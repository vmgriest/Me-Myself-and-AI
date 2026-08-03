# Unit tests for retrieval logic
#
# vector_store.search() and the embedding calls are mocked out here — this
# suite is testing retrieve_parent_context()'s own logic (deduping parents,
# preserving hit order, reading back parent text), not Qdrant or the
# embedding models themselves.
from unittest.mock import MagicMock

from src.rag_agent.nodes import retrieval


def _fake_hit(parent_id: str, score: float, source: str = "bio.md"):
    hit = MagicMock()
    hit.payload = {"parent_id": parent_id, "source": source}
    hit.score = score
    return hit


def test_retrieve_parent_context_dedupes_by_parent(monkeypatch):
    # two hits land in the same parent ("p1"), one in a different parent ("p2")
    hits = [_fake_hit("p1", score=0.9), _fake_hit("p1", score=0.7), _fake_hit("p2", score=0.5)]

    monkeypatch.setattr(retrieval.embeddings, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(retrieval.embeddings, "embed_sparse_query", lambda q: MagicMock())
    monkeypatch.setattr(retrieval.vector_store, "search", lambda client, dv, sv, k: hits)
    monkeypatch.setattr(retrieval.chunking, "load_parent", lambda pid: f"full text of {pid}")

    results = retrieval.retrieve_parent_context(client=MagicMock(), query="what did I study", k=4)

    assert len(results) == 2  # "p1" appears once despite two hits
    assert [r["parent_id"] for r in results] == ["p1", "p2"]  # order preserved, best score first
    assert results[0]["text"] == "full text of p1"
    assert results[0]["score"] == 0.9  # score from the FIRST (highest-scoring) hit for that parent, not overwritten


def test_retrieve_parent_context_empty_hits_returns_empty(monkeypatch):
    monkeypatch.setattr(retrieval.embeddings, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(retrieval.embeddings, "embed_sparse_query", lambda q: MagicMock())
    monkeypatch.setattr(retrieval.vector_store, "search", lambda client, dv, sv, k: [])

    results = retrieval.retrieve_parent_context(client=MagicMock(), query="anything", k=4)
    assert results == []


def test_make_retrieve_node_uses_rewritten_query(monkeypatch):
    captured_queries = []

    def fake_retrieve_parent_context(client, query, k):
        captured_queries.append(query)
        return [{"source": "bio.md", "text": "...", "score": 0.5, "parent_id": "p1"}]

    monkeypatch.setattr(retrieval, "retrieve_parent_context", fake_retrieve_parent_context)

    node = retrieval.make_retrieve_node(client=MagicMock())
    result = node({"rewritten_query": "standalone question", "messages": []})

    assert captured_queries == ["standalone question"]  # searched on the rewritten query, not the raw message
    assert result["retrieved_chunks"][0]["parent_id"] == "p1"


def test_make_retrieve_node_resets_web_search_state_for_new_turn(monkeypatch):
    # regression: web_results/used_web_search are plain (unreducered) state fields, so LangGraph's
    # checkpointer otherwise carries a previous turn's values straight into this one. retrieve() runs
    # first every turn, so it must explicitly reset both — otherwise a web search from an earlier,
    # unrelated question stays in state forever and gets silently reused on later turns.
    monkeypatch.setattr(retrieval, "retrieve_parent_context", lambda client, query, k: [])

    node = retrieval.make_retrieve_node(client=MagicMock())
    result = node({"rewritten_query": "a new unrelated question", "messages": []})

    assert result["web_results"] == []
    assert result["used_web_search"] is False
