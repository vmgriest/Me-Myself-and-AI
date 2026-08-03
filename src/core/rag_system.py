# Orchestrates retrieval + generation.
#
# Phase 3: answer() now runs the compiled LangGraph agent (graph.py) instead
# of Phase 1/2's plain function chain. The public answer() signature is
# unchanged apart from the added thread_id/model_choice params, so
# chat_interface.py only needed small updates (passing those through).
from src import config  # for get_llm() / default_model_choice()
from src.core import state_manager  # shared checkpointer + thread_id helpers
from src.core.execution_logger import traced  # names each answer() call as one run in LangSmith
from src.ingestion import vector_store  # to open the shared Qdrant client
from src.rag_agent.graph import build_graph  # compiles the LangGraph agent
from src.web.website_scraper import get_tavily_client  # shared client for the web-search fallback node


class RAGSystem:
    def __init__(self):
        # One Qdrant client for the app's lifetime, shared by every graph node via closure.
        self.client = vector_store.get_qdrant_client()
        self.tavily_client = get_tavily_client()  # None if TAVILY_API_KEY isn't configured — the node handles that
        self.checkpointer = state_manager.get_checkpointer()  # persists each thread's state for the process's lifetime
        # Compiled graphs are cached per model choice, since a graph's nodes capture one specific llm
        # instance via closure — switching models in the UI looks up (or lazily builds) a different
        # compiled graph rather than mutating the running one.
        self._graphs_by_model: dict[str, object] = {}

    def _get_graph(self, model_choice: str | None):
        model_choice = model_choice or config.default_model_choice()
        if model_choice not in self._graphs_by_model:
            llm = config.get_llm(model_choice)
            self._graphs_by_model[model_choice] = build_graph(llm, self.client, self.checkpointer, self.tavily_client)
        return self._graphs_by_model[model_choice]

    @traced
    def answer(self, query: str, thread_id: str, model_choice: str | None = None) -> dict:
        """Runs one turn of the agent for the given thread, using the given
        model choice (see config.list_available_models()). LangGraph merges
        this new human message into that thread's existing history (via the
        checkpointer), then the graph rewrites the query, retrieves
        parent-level context, falls back to a live web search if the docs
        didn't have the answer, and generates a grounded reply."""
        graph = self._get_graph(model_choice)
        thread_config = state_manager.get_thread_config(thread_id)  # tells the checkpointer which thread this is
        result = graph.invoke({"messages": [("human", query)]}, config=thread_config)  # runs the full graph once

        sources = list(result.get("retrieved_chunks", [])) + list(result.get("web_results", []))
        return {"answer": result["answer"], "sources": sources}
