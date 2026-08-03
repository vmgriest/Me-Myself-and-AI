# Retrieval logic: "small-to-big" — search on small child chunks (precise),
# but hand the LLM the larger parent chunk each hit belongs to (rich context).
#
# retrieve_parent_context() is the plain function used directly by
# rag_system.py. make_retrieve_node() wraps it as a LangGraph node, which
# searches on rewritten_query instead of the raw message.
from qdrant_client import QdrantClient  # type hint for the client passed in

from src import config  # for RETRIEVAL_TOP_K
from src.ingestion import chunking, embeddings, vector_store  # chunking.load_parent(), embeddings, vector_store.search()
from src.rag_agent.prompts import WEB_SEARCH_QUERY_PROMPT
from src.rag_agent.state import AgentState


def retrieve_parent_context(client: QdrantClient, query: str, k: int = config.RETRIEVAL_TOP_K) -> list[dict]:
    """Embeds the query (dense + sparse) and runs a hybrid top-k
    child-chunk search, then maps each hit to its parent chunk and returns
    the parent's full text instead of the tiny child snippet. Duplicate
    parents (multiple child hits from the same parent) are deduped so the
    LLM doesn't see the same context twice."""
    dense_vector = embeddings.embed_query(query)  # semantic embedding of the query
    sparse_vector = embeddings.embed_sparse_query(query)  # BM25/keyword embedding of the query
    hits = vector_store.search(client, dense_vector, sparse_vector, k=k)  # hybrid search, best match first

    seen_parent_ids: set[str] = set()  # tracks which parents we've already included
    results: list[dict] = []  # one entry per unique parent, in hit-score order

    for hit in hits:  # walk through child hits, best score first
        parent_id = hit.payload.get("parent_id")  # which parent this child chunk belongs to
        if parent_id in seen_parent_ids:  # already pulled in this parent from an earlier (higher-scoring) hit
            continue
        seen_parent_ids.add(parent_id)  # mark this parent as included

        parent_text = chunking.load_parent(parent_id)  # fetch the full parent text from the parent store
        results.append(
            {
                "source": hit.payload.get("source"),  # e.g. "bio.md", for UI attribution
                "text": parent_text,  # the larger parent text, not the small child snippet
                "score": hit.score,  # the child hit's fused (RRF) score, not a raw similarity value
                "parent_id": parent_id,  # for debugging/tracing
            }
        )

    return results


def make_retrieve_node(client: QdrantClient):
    """Builds a graph node that searches using rewritten_query (produced by
    the rewrite_query node) instead of the raw latest message, so follow-up
    questions retrieve correctly even when they rely on prior context."""

    def retrieve(state: AgentState) -> dict:
        results = retrieve_parent_context(client, state["rewritten_query"], k=config.RETRIEVAL_TOP_K)
        # web_results/used_web_search are plain (unreducered) state fields, so LangGraph's checkpointer
        # otherwise carries whatever value they had at the end of the *previous* turn straight into this
        # one — retrieve() runs first every turn, so this is where each new turn starts genuinely fresh.
        # Without this reset, a web search from an earlier, unrelated question stayed in state forever:
        # every later turn silently re-included that stale (possibly wrong-person) web content, and the
        # fallback could never fire again for a different question later in the same conversation.
        return {"retrieved_chunks": results, "web_results": [], "used_web_search": False}

    return retrieve


MAX_WEB_SEARCH_RESULTS = 4


def _build_search_query(llm, question: str) -> str:
    """Turns a natural-language question into a short, keyword-style query.
    Search APIs rank niche/specific pages (a sports-org tournament listing,
    a player profile) far better on keyword queries than on full sentences
    — "Vincent Griest " + a raw question missed real matches that "Vincent
    Griest USTA tennis player" found immediately at the top of results."""
    if llm is not None:
        try:
            response = llm.invoke(WEB_SEARCH_QUERY_PROMPT.format(question=question))
            query = response.content.strip().strip('"')
            if query:  # got something usable back
                return query
        except Exception as e:  # model error — fall through to the plain heuristic below
            print(f"web_search node: query-rewrite LLM call failed: {e}")

    # fallback (also used when no llm was given at all): same simple heuristic as before
    return question if "griest" in question.lower() else f"Vincent Griest {question}"


def make_web_search_node(tavily_client, llm=None):
    """Builds the fallback node the graph routes to when the ingested docs
    didn't have the answer (see generation.py's answer_indicates_no_info and
    graph.py's routing after "generate"). Runs a live Tavily web search
    instead of the local Qdrant index, so the agent can answer questions
    about things that were never ingested (e.g. news coverage of Vincent
    that isn't in his bio/resume/GitHub)."""

    def web_search(state: AgentState) -> dict:
        if tavily_client is None:  # TAVILY_API_KEY not configured — nothing to fall back to
            return {"web_results": [], "used_web_search": True}

        question = state["rewritten_query"]
        search_query = _build_search_query(llm, question)

        try:
            response = tavily_client.search(query=search_query, max_results=MAX_WEB_SEARCH_RESULTS, search_depth="advanced")
        except Exception as e:  # network error, bad key, rate limit — degrade gracefully, don't crash the turn
            print(f"web_search node: Tavily search failed: {e}")
            return {"web_results": [], "used_web_search": True}

        results = [
            {"source": r["url"], "text": r.get("content", ""), "score": r.get("score", 0.0)}
            for r in response.get("results", [])
            if r.get("content")  # skip results with no actual text
        ]
        return {"web_results": results, "used_web_search": True}  # flag set so the router doesn't loop back here again

    return web_search
