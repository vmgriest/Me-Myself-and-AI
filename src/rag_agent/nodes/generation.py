# Final answer aggregation & generation
import json  # to read the GitHub repo manifest
import re  # to strip the trailing GROUNDED: yes/no tag out of the displayed answer
from functools import lru_cache  # to only read bio.md/the repo manifest once per process

from langchain_core.messages import AIMessage  # the reply gets appended to state["messages"] as this type

from src import config
from src.rag_agent.prompts import GENERATION_SYSTEM_PROMPT, VERIFY_WEB_CLAIMS_PROMPT
from src.rag_agent.state import AgentState

# Matches the GROUNDED: yes/no tag wherever it appears, tolerating the formatting variations weaker/
# local models tend to introduce (markdown bold, trailing punctuation, not-quite-the-last-line) — an
# earlier version anchored strictly to the end of the string ($), which missed anything but an exact
# match and let the raw tag leak into the answer shown to the user.
GROUNDED_TAG_RE = re.compile(r"\n*\**\s*GROUNDED\s*:\s*(yes|no)\s*\.?\**", re.IGNORECASE)


@lru_cache(maxsize=1)
def _load_anchor() -> str:
    """A short, always-included block of verified facts (Vincent's own bio),
    used to sanity-check ambiguous claims — especially same-name collisions
    in live web search results — regardless of what retrieval happens to
    surface on a given turn. Cached since it's re-read on every generate
    call; bio.md doesn't change mid-process."""
    if not config.BIO_PATH.exists():  # not configured yet — degrade gracefully rather than crash every turn
        return "(no bio configured yet)"
    return config.BIO_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_repo_manifest() -> str:
    """A complete, always-included list of every ingested GitHub repo's name
    and description. "List/how many repos do you have" doesn't map well to
    vector search — there's no ranking or completeness signal in a top-k
    similarity search over README chunks, so such questions under-retrieved
    and fell back to a live web search that dragged in unrelated noise
    instead of just answering from what's already known. Cached like
    _load_anchor(); the manifest is only written once, at ingestion time."""
    if not config.GITHUB_REPOS_MANIFEST_PATH.exists():  # no GitHub source configured/ingested
        return ""
    manifest = json.loads(config.GITHUB_REPOS_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not manifest:
        return ""
    return "\n".join(f"- {repo['name']}: {repo['description']}" for repo in manifest)


# Fallback only: used when the model doesn't include the GROUNDED: tag
# (small/local models sometimes drop instructions like this). Pattern-matching
# English paraphrases is inherently incomplete — this list previously missed
# real refusals phrased as "doesn't *specifically* mention" or "does not
# *include information about*" — so the structured tag above is the primary
# signal graph.py's router uses; this is just a safety net.
NO_INFO_PHRASES = [
    "don't have that information",
    "don't have information",
    "does not mention",
    "doesn't mention",
    "no information about",
    "not mentioned in the context",
    "context does not",
    "doesn't specifically",
    "does not specifically",
    "not specify",
    "does not specify",
    "doesn't specify",
    "not specified",
    "no details about",
    "doesn't provide",
    "does not provide",
    "not provided in the context",
    "not available in the context",
    "unclear from the context",
    "i cannot find",
    "i couldn't find",
    "no mention of",
    "not found in the context",
]


def answer_indicates_no_info(answer: str) -> bool:
    lowered = answer.lower()  # case-insensitive match against NO_INFO_PHRASES
    return any(phrase in lowered for phrase in NO_INFO_PHRASES)


def make_generate_node(llm):
    """Builds the graph node that produces the final answer from whatever
    retrieve_parent_context() found (plus any live web_results, once the
    web_search fallback node has run), the rewritten question, and the
    running conversation summary."""

    def generate_answer(state: AgentState) -> dict:
        chunks = state.get("retrieved_chunks", [])  # parent-level context from the retrieve node
        context_parts = [c["text"] for c in chunks]  # docs' own context first

        repo_manifest = _load_repo_manifest()
        if repo_manifest:  # complete list, independent of what vector search happened to retrieve this turn
            context_parts.append(f"Complete list of Vincent's GitHub repositories (use this for any listing/counting question):\n{repo_manifest}")

        web_results = state.get("web_results") or []
        if web_results:  # only present once the web_search fallback node has actually run
            web_text = "\n\n".join(f"{r['source']}: {r['text']}" for r in web_results)
            context_parts.append(f"From a live web search (use this if the above doesn't answer the question):\n{web_text}")

        context = "\n\n---\n\n".join(context_parts)  # join everything for the prompt

        system_prompt = GENERATION_SYSTEM_PROMPT.format(
            summary=state.get("summary") or "(none yet)",  # keeps earlier-turn context even after messages are trimmed
            anchor=_load_anchor(),  # stable ground truth to check ambiguous/conflicting claims against
            context=context,
        )
        prior_turns = state["messages"][:-1]  # earlier turns in this thread, excluding the just-asked raw question
        # system prompt (grounding + retrieved context), then actual conversation history, then the
        # standalone rewritten question as the final turn — without prior_turns here, the LLM would have
        # no way to recall anything from earlier in the conversation that wasn't re-surfaced by retrieval.
        messages = [("system", system_prompt), *prior_turns, ("human", state["rewritten_query"])]
        response = llm.invoke(messages)  # call the LLM for the final answer
        raw = response.content

        tag_match = GROUNDED_TAG_RE.search(raw)
        if tag_match:  # the model included the structured tag — trust it, and cut it out wherever it landed
            answer_text = (raw[: tag_match.start()] + raw[tag_match.end() :]).strip()
            grounded = tag_match.group(1).lower() == "yes"
        else:  # model forgot the tag — fall back to the (less reliable) phrase heuristic
            answer_text = raw.rstrip()
            grounded = not answer_indicates_no_info(answer_text)

        # last-resort safety net: if a malformed/repeated tag still slipped through the regex above,
        # never show raw "GROUNDED" bookkeeping text to the user — drop any line containing it.
        if "grounded" in answer_text.lower():
            answer_text = "\n".join(
                line for line in answer_text.splitlines() if "grounded" not in line.lower()
            ).strip()

        # returned as a new AIMessage so add_messages appends it to the persisted thread history
        return {"answer": answer_text, "grounded": grounded, "messages": [AIMessage(content=answer_text)]}

    return generate_answer


# If the "corrected" answer talks about the verification process itself instead of just answering,
# something went wrong (the verify LLM got confused, usually because a weaker/local model ignored the
# "never comment on the verification process" instruction) — better to keep the original draft than
# show the user a broken meta-response like "I couldn't verify any of the claims in your draft answer".
VERIFY_SELF_REFERENCE_PHRASES = [
    "draft answer",
    "couldn't verify",
    "could not verify",
    "unable to verify",
    "verification process",
    "web search results provided",
    "provided web search results",
]


def _looks_like_broken_verify_output(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in VERIFY_SELF_REFERENCE_PHRASES)


def make_verify_node(llm):
    """Builds a second, independent check that runs only on turns where the
    web-search fallback was used — the highest-risk path for the same-name-
    collision problem, since Tavily can return real pages about a different
    person entirely. Re-checks the draft answer's specific claims against
    the raw web snippets, local docs, and verified facts, and rewrites out
    anything not clearly supported, rather than trusting generate_answer's
    single pass. Falls back to the original draft if the check itself
    produces a broken response, rather than showing that to the user."""

    def verify(state: AgentState) -> dict:
        web_results = state.get("web_results") or []
        if not web_results:  # nothing web-sourced to double-check this turn — generate_answer's pass stands as-is
            return {}

        draft = state["answer"]  # generate_answer's output, to be checked and possibly corrected
        # local retrieved docs too, not just web results — a draft is often grounded in BOTH at once
        # (e.g. "list your repos" is entirely local); without this, verify judged local-doc claims
        # against the wrong source and concluded it couldn't confirm anything that was actually fine
        local_context = "\n\n---\n\n".join(c["text"] for c in state.get("retrieved_chunks", []))
        web_text = "\n\n".join(f"{r['source']}: {r['text']}" for r in web_results)  # the raw web snippets to check against
        prompt = VERIFY_WEB_CLAIMS_PROMPT.format(
            anchor=_load_anchor(),
            local_context=local_context or "(none retrieved this turn)",
            web_text=web_text,
            draft=draft,
        )
        response = llm.invoke(prompt)  # independent second call — not the same call re-confirming its own first take
        corrected = response.content.strip()

        if not corrected or _looks_like_broken_verify_output(corrected):
            corrected = draft  # verification itself misfired — show the original answer, not a broken one

        last_message = state["messages"][-1]  # the AIMessage generate_answer() just appended this turn
        # same id as the message being replaced — add_messages overwrites by id instead of duplicating
        return {"answer": corrected, "messages": [AIMessage(content=corrected, id=last_message.id)]}

    return verify
