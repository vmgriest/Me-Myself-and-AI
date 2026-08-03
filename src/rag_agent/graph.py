# Defines the LangGraph workflow nodes & edges
from langgraph.graph import END, START, StateGraph  # graph builder + the two special "boundary" nodes
from qdrant_client import QdrantClient  # type hint for the client passed in

from src.rag_agent.nodes.conversation import SUMMARIZE_AFTER_N_MESSAGES, make_rewrite_query_node, make_summarize_node
from src.rag_agent.nodes.generation import make_generate_node, make_verify_node
from src.rag_agent.nodes.retrieval import make_retrieve_node, make_web_search_node
from src.rag_agent.state import AgentState


def _should_summarize(state: AgentState) -> str:
    """Router evaluated right after START: only pay for the extra
    summarization LLM call once conversation history actually gets long,
    otherwise skip straight to rewriting the query."""
    if len(state["messages"]) > SUMMARIZE_AFTER_N_MESSAGES:  # history has grown past the threshold
        return "summarize"  # go fold older turns into the running summary first
    return "rewrite_query"  # history still short, skip summarization this turn


def _route_after_generate(state: AgentState) -> str:
    """Router evaluated right after "generate":
    - If this pass used live web search results, always verify them before
      finishing — that's the highest-risk path for the same-name-collision
      problem (a real but unrelated person sharing Vincent's name).
    - Else if the docs didn't have the answer (per generate_answer's own
      GROUNDED: yes/no self-report) and we haven't already tried a web
      search this turn, fall back to one and regenerate.
    - Otherwise the turn is done. The used_web_search guard prevents
      looping forever on a question the web search also can't answer."""
    if state.get("web_results"):  # this pass used web content — check its claims before finishing
        return "verify"
    if state.get("used_web_search"):  # already tried web search this turn and it found nothing usable
        return "end"
    if not state.get("grounded", True):  # docs didn't have it — worth trying a live search
        return "web_search"
    return "end"  # docs had the answer — no need to search the web


def build_graph(llm, client: QdrantClient, checkpointer, tavily_client=None):
    """Wires conversation -> retrieval -> generation into a compiled graph,
    with a conditional fallback to a live web search when the ingested docs
    don't have the answer, and a verification pass on any web-sourced
    answer before it's shown. Persistence is per-thread via the given
    checkpointer (see core/state_manager.py) so conversations survive
    across turns."""
    graph = StateGraph(AgentState)  # new graph builder, typed to our state schema

    graph.add_node("summarize", make_summarize_node(llm))  # folds older turns into state["summary"]
    graph.add_node("rewrite_query", make_rewrite_query_node(llm))  # produces a standalone version of the question
    graph.add_node("retrieve", make_retrieve_node(client))  # small-to-big retrieval using the rewritten query; also resets web_results/used_web_search for this turn
    graph.add_node("generate", make_generate_node(llm))  # produces the final answer
    graph.add_node("web_search", make_web_search_node(tavily_client, llm))  # live search fallback when docs come up empty
    graph.add_node("verify", make_verify_node(llm))  # independent re-check of any web-sourced claims

    # from START, branch to "summarize" or straight to "rewrite_query" depending on history length
    graph.add_conditional_edges(START, _should_summarize, {"summarize": "summarize", "rewrite_query": "rewrite_query"})
    graph.add_edge("summarize", "rewrite_query")  # after summarizing, always continue to rewriting the query
    graph.add_edge("rewrite_query", "retrieve")  # then retrieve context for the (possibly rewritten) question
    graph.add_edge("retrieve", "generate")  # then generate the answer from that context
    # after generating: verify web-sourced claims, fall back to a live search, or end the turn
    graph.add_conditional_edges("generate", _route_after_generate, {"verify": "verify", "web_search": "web_search", "end": END})
    graph.add_edge("web_search", "generate")  # regenerate using the freshly-fetched web context
    graph.add_edge("verify", END)  # verification always finishes the turn — no further looping

    return graph.compile(checkpointer=checkpointer)  # compiled, runnable graph with per-thread persistence
