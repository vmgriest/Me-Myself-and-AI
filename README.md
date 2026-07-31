# My Personal AI

An "About Me" personal assistant: an agentic RAG system built on your own bio, GitHub repos, and websites, using LangGraph for orchestration, Qdrant for hybrid (dense + sparse) retrieval, and Gradio for the chat UI.

## Project Structure

```
data/            Raw sources, processed chunks, and parent-chunk store
qdrant_db/       Local Qdrant vector database files
src/
  core/          Orchestration, chat interface, logging, state management
  rag_agent/     LangGraph workflow: graph, state, nodes, tools, prompts
  ingestion/     Document loading, chunking, embeddings, vector store
  web/           GitHub and website connectors
  utils/         Shared helpers
notebooks/       Tutorial and experimentation notebooks
tests/           Unit and integration tests
docker/          Docker Compose for Qdrant + Ollama
```

## Core Technologies

| Component | Tech |
|---|---|
| LLM | Ollama (local dev) / Gemini or OpenAI (production) |
| Vector DB | Qdrant (local, file-based, hybrid search) |
| Embeddings | Dense: `all-mpnet-base-v2`, Sparse: BM25 |
| Framework | LangGraph + LangChain |
| Frontend | Gradio |
| Containerization | Docker Compose |

## Setup (planned)

1. `python -m venv .venv` and activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in values
4. `docker compose -f docker/docker-compose.yml up -d` to start Qdrant + Ollama
5. Add source material to `data/raw/`
6. Run ingestion, then `python src/main.py` to launch the chat UI

This is scaffolding only — implementation to follow.
