# Logging and observability (LangSmith tracing)
#
# LangChain/LangGraph auto-instrument every LLM call and graph run via the
# LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT env vars in
# .env (loaded by config.py) — no manual callback-handler wiring is needed.
# @traceable below just gives the top-level run a readable name in the
# LangSmith UI; it's a no-op if tracing isn't enabled, so nothing here
# breaks when LANGCHAIN_API_KEY is unset.
import os  # to read the tracing env vars for the startup status message

from langsmith import traceable  # decorator that names/groups a run in LangSmith, no-ops if tracing is off


def tracing_enabled() -> bool:
    # true only if the user has actually opted in (matches what LangChain itself checks)
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and bool(os.getenv("LANGCHAIN_API_KEY"))


def log_startup_status() -> None:
    """Prints whether tracing is active, so it's obvious at a glance when
    running the app whether calls are being sent to LangSmith."""
    if tracing_enabled():  # both the flag and a key are present
        project = os.getenv("LANGCHAIN_PROJECT", "default")  # which LangSmith project runs will show up under
        print(f"LangSmith tracing enabled (project: {project})")
    else:  # not configured — the app still runs fine, just untraced
        print("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env to enable)")


# applied to RAGSystem.answer in rag_system.py so each turn shows up as one named run in LangSmith
traced = traceable(name="rag_system.answer", run_type="chain")
