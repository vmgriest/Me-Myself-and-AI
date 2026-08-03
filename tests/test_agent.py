# Unit tests for the LangGraph agent
#
# Uses a fake LLM (no real Ollama/Groq calls) and a MemorySaver checkpointer,
# so this suite runs offline and fast. Covers the two node-level behaviors
# with real subtlety (rewrite_query's early-return, summarize's threshold +
# trimming) and that the graph actually wires together into a runnable
# whole.
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver

from src.rag_agent import graph as graph_module
from src.rag_agent.nodes.conversation import (
    KEEP_LAST_N_MESSAGES,
    SUMMARIZE_AFTER_N_MESSAGES,
    make_rewrite_query_node,
    make_summarize_node,
)
from src.rag_agent.nodes.generation import answer_indicates_no_info, make_generate_node, make_verify_node
from src.rag_agent.nodes.retrieval import make_web_search_node


def _fake_llm(response_text: str = "response"):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=response_text)
    return llm


def test_rewrite_query_skips_llm_on_first_turn():
    llm = _fake_llm()
    node = make_rewrite_query_node(llm)

    state = {"messages": [HumanMessage(content="Where did I go to college?")], "summary": None}
    result = node(state)

    llm.invoke.assert_not_called()  # no prior context to resolve against, so no LLM call needed
    assert result["rewritten_query"] == "Where did I go to college?"


def test_rewrite_query_calls_llm_when_history_exists():
    llm = _fake_llm("standalone rewritten question")
    node = make_rewrite_query_node(llm)

    state = {
        "messages": [
            HumanMessage(content="Where did I go to college?"),
            AIMessage(content="Oakland University."),
            HumanMessage(content="What year did I graduate from there?"),
        ],
        "summary": None,
    }
    result = node(state)

    llm.invoke.assert_called_once()
    assert result["rewritten_query"] == "standalone rewritten question"


def test_summarize_node_noop_below_threshold():
    llm = _fake_llm()
    node = make_summarize_node(llm)

    messages = [HumanMessage(content="hi", id=str(i)) for i in range(SUMMARIZE_AFTER_N_MESSAGES - 1)]
    result = node({"messages": messages, "summary": None})

    llm.invoke.assert_not_called()
    assert result == {}  # no state changes below the threshold


def test_summarize_node_trims_and_summarizes_above_threshold():
    llm = _fake_llm("a concise summary")
    node = make_summarize_node(llm)

    messages = [HumanMessage(content=f"msg {i}", id=str(i)) for i in range(SUMMARIZE_AFTER_N_MESSAGES + 2)]
    result = node({"messages": messages, "summary": None})

    llm.invoke.assert_called_once()
    assert result["summary"] == "a concise summary"

    removed_ids = {m.id for m in result["messages"] if isinstance(m, RemoveMessage)}
    expected_removed_ids = {m.id for m in messages[:-KEEP_LAST_N_MESSAGES]}  # everything except the last N
    assert removed_ids == expected_removed_ids


def test_should_summarize_routes_by_message_count():
    short_state = {"messages": [HumanMessage(content="hi", id="1")]}
    assert graph_module._should_summarize(short_state) == "rewrite_query"

    long_messages = [HumanMessage(content="hi", id=str(i)) for i in range(SUMMARIZE_AFTER_N_MESSAGES + 1)]
    assert graph_module._should_summarize({"messages": long_messages}) == "summarize"


def test_build_graph_compiles_with_expected_nodes():
    llm = _fake_llm()
    client = MagicMock()
    checkpointer = MemorySaver()

    compiled = graph_module.build_graph(llm, client, checkpointer)

    node_names = set(compiled.get_graph().nodes.keys())
    assert {"summarize", "rewrite_query", "retrieve", "generate", "web_search", "verify"}.issubset(node_names)


def test_answer_indicates_no_info_matches_expected_phrases():
    assert answer_indicates_no_info("I don't have that information about his hobbies.")
    assert answer_indicates_no_info("The context does not mention any sports.")
    assert not answer_indicates_no_info("Vincent works as a Software Engineer at TCS.")


def test_route_after_generate_falls_back_once_then_stops():
    ungrounded_state = {"grounded": False, "used_web_search": False, "web_results": []}
    assert graph_module._route_after_generate(ungrounded_state) == "web_search"

    already_tried_state = {"grounded": False, "used_web_search": True, "web_results": []}
    assert graph_module._route_after_generate(already_tried_state) == "end"  # web search ran and found nothing usable

    grounded_state = {"grounded": True, "used_web_search": False, "web_results": []}
    assert graph_module._route_after_generate(grounded_state) == "end"  # docs already had the answer


def test_route_after_generate_verifies_when_web_results_present():
    # any turn where web content was actually used gets checked before finishing, regardless of
    # whether generate_answer itself already reported grounded — verify is an independent check
    state = {"grounded": True, "used_web_search": True, "web_results": [{"source": "x", "text": "y", "score": 0.5}]}
    assert graph_module._route_after_generate(state) == "verify"


def test_web_search_node_returns_empty_when_tavily_not_configured():
    node = make_web_search_node(tavily_client=None)
    result = node({"rewritten_query": "does Vincent play sports"})
    assert result == {"web_results": [], "used_web_search": True}


def test_web_search_node_formats_tavily_results():
    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [
            {"url": "https://example.com/a", "content": "Vincent plays tennis.", "score": 0.9},
            {"url": "https://example.com/b", "content": "", "score": 0.1},  # no content — should be skipped
        ]
    }
    node = make_web_search_node(fake_client)

    result = node({"rewritten_query": "does he play sports"})  # doesn't mention "Vincent" by name

    called_query = fake_client.search.call_args.kwargs["query"]
    assert "Vincent Griest" in called_query  # query gets biased toward the right person

    assert result["used_web_search"] is True
    assert len(result["web_results"]) == 1  # the empty-content result was filtered out
    assert result["web_results"][0]["source"] == "https://example.com/a"


def test_web_search_node_enriches_first_name_only_queries():
    # regression: "vincent" alone used to be treated as already-disambiguated, letting queries like
    # this one search ambiguously and surface unrelated famous Vincents (Vince Carter, Gabe Vincent, ...)
    fake_client = MagicMock()
    fake_client.search.return_value = {"results": []}
    node = make_web_search_node(fake_client)

    node({"rewritten_query": "What sports did Vincent play in school?"})

    called_query = fake_client.search.call_args.kwargs["query"]
    assert "Griest" in called_query  # surname added even though the first name was already present


def test_web_search_node_uses_llm_to_build_keyword_query():
    fake_client = MagicMock()
    fake_client.search.return_value = {"results": []}
    llm = _fake_llm("Vincent Griest USTA tennis player")  # the exact phrasing that scored 0.99 in manual testing

    node = make_web_search_node(fake_client, llm)
    node({"rewritten_query": "What sports did Vincent play in school?"})

    llm.invoke.assert_called_once()
    called_query = fake_client.search.call_args.kwargs["query"]
    assert called_query == "Vincent Griest USTA tennis player"  # used the LLM's rewritten query, not the raw heuristic


def test_web_search_node_falls_back_to_heuristic_when_llm_fails():
    fake_client = MagicMock()
    fake_client.search.return_value = {"results": []}
    broken_llm = MagicMock()
    broken_llm.invoke.side_effect = RuntimeError("model unavailable")

    node = make_web_search_node(fake_client, broken_llm)
    node({"rewritten_query": "What sports did Vincent play?"})

    called_query = fake_client.search.call_args.kwargs["query"]
    assert "Griest" in called_query  # degraded to the plain heuristic instead of crashing the turn


def _generate_state(rewritten_query: str = "What is your job?"):
    return {
        "messages": [HumanMessage(content=rewritten_query)],
        "rewritten_query": rewritten_query,
        "retrieved_chunks": [{"text": "Vincent works at TCS.", "source": "bio.md", "score": 0.8, "parent_id": "p0"}],
        "summary": None,
    }


def test_load_repo_manifest_formats_as_bullet_list(tmp_path, monkeypatch):
    import json

    from src import config
    from src.rag_agent.nodes import generation

    manifest_path = tmp_path / "github_repos.json"
    manifest_path.write_text(json.dumps([{"name": "RepoOne", "description": "does a thing"}]), encoding="utf-8")
    monkeypatch.setattr(config, "GITHUB_REPOS_MANIFEST_PATH", manifest_path)
    generation._load_repo_manifest.cache_clear()  # lru_cache — clear so this test doesn't see a stale result

    assert generation._load_repo_manifest() == "- RepoOne: does a thing"
    generation._load_repo_manifest.cache_clear()  # don't leak this test's cached value into other tests


def test_load_repo_manifest_empty_when_not_configured(tmp_path, monkeypatch):
    from src import config
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(config, "GITHUB_REPOS_MANIFEST_PATH", tmp_path / "does_not_exist.json")
    generation._load_repo_manifest.cache_clear()

    assert generation._load_repo_manifest() == ""
    generation._load_repo_manifest.cache_clear()


def test_generate_answer_includes_repo_manifest_in_prompt(monkeypatch):
    # regression: "list/how many repos" questions don't map well to vector search (no ranking or
    # completeness signal in a top-k similarity search), so they under-retrieved and fell back to a
    # web search that dragged in unrelated noise — always including the full repo list sidesteps that
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(generation, "_load_anchor", lambda: "fake anchor facts")
    monkeypatch.setattr(generation, "_load_repo_manifest", lambda: "- RepoOne: does a thing\n- RepoTwo: does another thing")
    llm = _fake_llm("Vincent has two repos.\nGROUNDED: yes")
    node = make_generate_node(llm)

    node(_generate_state("List his repos"))

    sent_messages = llm.invoke.call_args[0][0]
    system_prompt = sent_messages[0][1]  # ("system", prompt) tuple
    assert "RepoOne" in system_prompt
    assert "RepoTwo" in system_prompt


def test_generate_answer_parses_grounded_yes_tag_and_strips_it(monkeypatch):
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(generation, "_load_anchor", lambda: "fake anchor facts")  # no real bio.md dependency in tests
    llm = _fake_llm("Vincent works at TCS.\nGROUNDED: yes")
    node = make_generate_node(llm)

    result = node(_generate_state())

    assert result["grounded"] is True
    assert result["answer"] == "Vincent works at TCS."  # the tag line is stripped from what's shown to the user
    assert "GROUNDED" not in result["messages"][0].content  # and from what's persisted to history


def test_generate_answer_parses_grounded_no_tag(monkeypatch):
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(generation, "_load_anchor", lambda: "fake anchor facts")
    llm = _fake_llm("I don't have that information.\nGROUNDED: no")
    node = make_generate_node(llm)

    result = node(_generate_state())

    assert result["grounded"] is False
    assert result["answer"] == "I don't have that information."


def test_generate_answer_strips_grounded_tag_formatting_variations(monkeypatch):
    # regression: weaker/local models don't always emit the exact "GROUNDED: yes" at the very end —
    # a strict end-anchored regex missed these and let the raw tag leak into the visible answer
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(generation, "_load_anchor", lambda: "fake anchor facts")
    variations = [
        "Vincent works at TCS.\n**GROUNDED: yes**",  # markdown bold
        "Vincent works at TCS.\nGROUNDED: Yes.",  # trailing period
        "Vincent works at TCS.\nGROUNDED:yes",  # no space after colon
        "Vincent works at TCS.\nGROUNDED: yes\n",  # trailing newline after the tag
    ]
    for raw in variations:
        llm = _fake_llm(raw)
        node = make_generate_node(llm)
        result = node(_generate_state())

        assert result["grounded"] is True, f"failed to parse: {raw!r}"
        assert "grounded" not in result["answer"].lower(), f"tag leaked for: {raw!r}"
        assert result["answer"] == "Vincent works at TCS."


def test_generate_answer_safety_net_strips_malformed_tag_line(monkeypatch):
    # even if the regex somehow misses a malformed tag entirely, no line containing "grounded"
    # should ever reach the user
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(generation, "_load_anchor", lambda: "fake anchor facts")
    llm = _fake_llm("Vincent works at TCS.\n[[[ GROUNDED status: yes, confidence high ]]]")
    node = make_generate_node(llm)

    result = node(_generate_state())

    assert "grounded" not in result["answer"].lower()
    assert result["answer"] == "Vincent works at TCS."


def test_generate_answer_falls_back_to_phrase_heuristic_when_tag_missing(monkeypatch):
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(generation, "_load_anchor", lambda: "fake anchor facts")
    # exactly the real-world phrasing that the old phrase-only heuristic missed
    llm = _fake_llm("The context doesn't specifically mention the sports Vincent played in school.")
    node = make_generate_node(llm)

    result = node(_generate_state())

    assert result["grounded"] is False  # caught by the broadened fallback phrase list
    assert result["answer"] == "The context doesn't specifically mention the sports Vincent played in school."


def test_verify_node_noop_when_no_web_results():
    llm = _fake_llm("should not be called")
    node = make_verify_node(llm)

    state = {"web_results": [], "answer": "Vincent works at TCS.", "messages": [AIMessage(content="Vincent works at TCS.")]}
    result = node(state)

    llm.invoke.assert_not_called()
    assert result == {}  # nothing web-sourced this turn — nothing to verify


def test_verify_node_corrects_unsupported_claim_and_overwrites_message_by_id(monkeypatch):
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(generation, "_load_anchor", lambda: "Vincent attends Oakland University.")
    corrected_text = "Vincent plays tennis. I found a mention of a tennis player at a different school but couldn't confirm it's him."
    llm = _fake_llm(corrected_text)
    node = make_verify_node(llm)

    original_message = AIMessage(content="Vincent plays tennis and was on a different school's team.", id="msg-1")
    state = {
        "web_results": [{"source": "https://example.com", "text": "A Vincent Griest played tennis at Some Other School.", "score": 0.5}],
        "answer": "Vincent plays tennis and was on a different school's team.",
        "messages": [original_message],
    }
    result = node(state)

    llm.invoke.assert_called_once()
    assert result["answer"] == corrected_text
    assert result["messages"][0].id == "msg-1"  # same id as the original — overwrites rather than duplicates


def test_verify_node_includes_local_docs_in_its_prompt(monkeypatch):
    # regression: verify() used to only see web_results, so a draft grounded in LOCAL docs (e.g.
    # "list your GitHub repos") got judged against the wrong source entirely and always failed
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(generation, "_load_anchor", lambda: "fake anchor facts")
    llm = _fake_llm("Vincent's repos are X, Y, and Z.")
    node = make_verify_node(llm)

    state = {
        "web_results": [{"source": "https://example.com", "text": "unrelated web snippet", "score": 0.2}],
        "retrieved_chunks": [{"text": "GitHub repo: X. GitHub repo: Y. GitHub repo: Z.", "source": "github:X", "score": 0.8, "parent_id": "p0"}],
        "answer": "Vincent's repos are X, Y, and Z.",
        "messages": [AIMessage(content="Vincent's repos are X, Y, and Z.", id="msg-1")],
    }
    node(state)

    prompt_sent = llm.invoke.call_args[0][0]
    assert "GitHub repo: X" in prompt_sent  # the local doc content actually reached the verify prompt


def test_verify_node_falls_back_to_draft_when_output_looks_broken(monkeypatch):
    # regression: a real run produced "I couldn't verify any of the claims in your draft answer..."
    # as the FINAL answer shown to the user — a broken meta-response, not an actual answer
    from src.rag_agent.nodes import generation

    monkeypatch.setattr(generation, "_load_anchor", lambda: "fake anchor facts")
    broken_output = "I couldn't verify any of the claims in your draft answer against the provided web search results."
    llm = _fake_llm(broken_output)
    node = make_verify_node(llm)

    draft = "Vincent's top repos include URL Shortener and SpoiledOrNot."
    state = {
        "web_results": [{"source": "https://example.com", "text": "irrelevant", "score": 0.1}],
        "retrieved_chunks": [],
        "answer": draft,
        "messages": [AIMessage(content=draft, id="msg-1")],
    }
    result = node(state)

    assert result["answer"] == draft  # kept the original instead of surfacing the broken response
    assert result["messages"][0].content == draft
