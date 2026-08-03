# Entry point to run the Gradio web app
#
# Run from the project root with: python -m src.main
# (Requires ingestion to have already been run at least once — see
# src/ingestion/pipeline.py — and Ollama running locally.)
from src.core.chat_interface import build_interface  # builds the Gradio UI
from src.core.execution_logger import log_startup_status  # prints whether LangSmith tracing is active
from src.core.rag_system import RAGSystem  # the backend that answers questions


def main():
    log_startup_status()  # so it's obvious at a glance whether this run is being traced
    rag_system = RAGSystem()  # sets up the Qdrant client + LLM once
    interface = build_interface(rag_system)  # wire the backend into the chat UI
    interface.launch()  # starts the local web server (default http://127.0.0.1:7860)


if __name__ == "__main__":  # only runs when executed directly (python -m src.main)
    main()
