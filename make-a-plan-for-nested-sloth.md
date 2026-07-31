# Implementation Plan: my-personal-ai

## Context

The project scaffolding (folders + one-line stub comments) already exists at the repo root. Nothing is implemented yet. This plan sequences the actual build so that a working system exists as early as possible, then grows in capability — rather than fully building every module from the original architecture diagram before anything runs end-to-end.

Confirmed decisions (from user Q&A):
- **LLM**: start on local Ollama; design an `LLM_PROVIDER`-driven factory so switching to Gemini/OpenAI later is a config change, not a rewrite.
- **Qdrant**: embedded/local file mode (`QdrantClient(path=...)`), no Docker server for dev.
- **Data sources**: build ingestion generically, but the first working version ingests `bio.md` only. GitHub + website connectors come later.
- **Build order**: walking skeleton first (thinnest end-to-end slice that actually answers questions), then layer in parent-child chunking, hybrid search, the full LangGraph agent, extra connectors, observability, and Docker.

Verified environment facts that affect setup:
- Python 3.11.9 and Ollama are already installed and on PATH.
- Ollama has `codellama`, `llama3.2`, `llama3:8b`, `gemma4:26b`, `glm-4.7-flash` pulled — **not** `qwen3:4b`, which `.env.example` currently defaults to. Either `ollama pull qwen3:4b` or repoint `.env` at `llama3.2` (smallest already-available model) for the fastest first run.
- An NVIDIA GPU is present, which matters for the torch-install gotcha in Phase 0.
- The repo path contains a space (`Coding Practice`) — a latent gotcha for Docker volume mounts in Phase 5; not an issue before then.
- `data/raw/bio.md` is still an empty placeholder and must be filled with real content before Phase 1's smoke test means anything.

---

## Phase 0 — Environment Setup

1. `python -m venv .venv`; activate (PowerShell: `.venv\Scripts\Activate.ps1`; Git Bash: `source .venv/Scripts/activate`).
2. Install torch CPU-only first to avoid pip defaulting to a multi-GB CUDA wheel for a small personal-bio corpus: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`. GPU remains available later if large-scale GitHub ingestion (Phase 4) warrants faster embedding — a one-line change then.
3. `fastembed` (sparse/BM25) isn't needed until Phase 4 — fine to leave in requirements.txt, but its `onnxruntime` download can be skipped for the fastest Phase 1 install if desired.
4. Confirm Ollama is running (`ollama serve` / tray app) before the app starts — `ChatOllama` fails fast with a connection error otherwise.
5. Copy `.env.example` → `.env`; set `OLLAMA_MODEL=llama3.2` (already pulled) or `ollama pull qwen3:4b` first; keep `QDRANT_PATH=./qdrant_db`, `QDRANT_COLLECTION=personal_ai`.
6. Write real content into `data/raw/bio.md` — required for a meaningful Phase 1 smoke test.

---

## Phase 1 — Walking Skeleton

Goal: `bio.md` → chunk → embed → Qdrant → retrieve → generate → Gradio chat, with **no LangGraph and no hierarchical chunking yet** — those are premature complexity for this phase.

Build order (each step depends on the previous):

1. **`src/config.py`** — `load_dotenv()`; path constants via `pathlib.Path` anchored to project root (don't hardcode backslashes, given the space in the parent directory). Model/setting constants (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `QDRANT_PATH`, `QDRANT_COLLECTION`, `DENSE_EMBED_MODEL = "all-mpnet-base-v2"`).
   - **LLM factory**: `get_llm()` reads `LLM_PROVIDER` (default `"ollama"`) → returns `ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)`. Add guarded `elif` branches for `"gemini"`/`"openai"` with lazy imports and a clear error if selected without the package/API key present — this is what makes the later provider swap a one-line change.

2. **`src/ingestion/document_loader.py`** — `load_bio()` reads `data/raw/bio.md` into a `Document` with `metadata={"source": "bio.md", "source_type": "bio"}`. Wrap it in `load_all_sources()` (currently just calling `load_bio()`) so Phase 4 can add GitHub/website branches without restructuring callers.

3. **`src/ingestion/chunking.py`** — `simple_chunk(documents)` via `RecursiveCharacterTextSplitter` (`chunk_size≈500-800`, `overlap≈100`), stable `chunk_id` per chunk in metadata. Keep the `list[Document] in → list[Document] out` interface stable — Phase 2 replaces the internals, not the shape.

4. **`src/ingestion/embeddings.py`** — dense embedder wrapping `sentence-transformers` (`all-mpnet-base-v2`), `normalize_embeddings=True` (needed for correct cosine distance in Qdrant). First call downloads ~420MB to the HF cache — worth pre-warming standalone before running full ingestion the first time.

5. **`src/ingestion/vector_store.py`** — `get_qdrant_client()` → `QdrantClient(path=QDRANT_PATH)`; `ensure_collection()` (`VectorParams(size=768, distance=Distance.COSINE)`); `upsert_chunks()`; `search()` via `client.query_points(...)` (current qdrant-client API — `search()` is deprecated).
   - **Gotcha**: local file mode holds an exclusive lock on `QDRANT_PATH`. Ingestion and the running Gradio app can't hold the client open at the same time — run ingestion to completion and exit before launching `src/main.py`.

6. **`src/ingestion/pipeline.py`** *(new file — justified: no existing stub owns "run ingestion end-to-end"; `document_loader.py` only loads, `main.py` is explicitly the Gradio entry point)* — `run_ingestion()` chains load → chunk → embed → `ensure_collection()` → `upsert_chunks()`, runnable via `python -m src.ingestion.pipeline`. This becomes the single seam Phases 2 and 4 extend.

7. **`src/core/rag_system.py`** — Phase 1 implements this as a **plain function chain**, not LangGraph (despite the stub comment — that's the Phase 3 target). Public method `answer(query: str) -> dict` returning `{"answer": ..., "sources": [...]}`: embed query → `vector_store.search()` → build prompt from context chunks → `config.get_llm().invoke(...)`. Keep this signature stable so Phase 3 can swap internals to `graph.invoke()` without touching callers.

8. **`src/core/chat_interface.py`** — `build_interface(rag_system)` using `gr.ChatInterface(fn=..., type="messages")`, rendering the answer plus a compact view of source chunks so grounding is visually verifiable.

9. **`src/main.py`** — wire config → `RAGSystem` → interface → `.launch()`.

**Verification**: `python -m src.ingestion.pipeline` completes and `qdrant_db/` grows on disk; `python src/main.py` then answers a bio-grounded question correctly with the right source chunk shown, and doesn't hallucinate confidently on an out-of-scope question (GitHub/website content isn't ingested yet, so "I don't have that" is fine).

---

## Phase 2 — Hierarchical Parent-Child Chunking

1. **`src/ingestion/chunking.py`** (extend) — `parent_child_chunk()`: split into large parent chunks (~1500-2500 chars, roughly section-sized, stable `parent_id`) then small child chunks (~300-500 chars, metadata carries `parent_id`). `save_parents()`/`load_parent(parent_id)` write/read `data/parent_store/` (JSONL or one file per id).
   - **Mapping**: `data/processed/` = full intermediate chunk records at both levels (debugging/re-embed artifact); `data/parent_store/` = parent chunks only, read at query time; **Qdrant** = child-chunk vectors + metadata (`text`, `source`, `parent_id`) only — never full parent text, to keep the index lean.

2. **`src/ingestion/pipeline.py`** — swap `simple_chunk` for `parent_child_chunk`; embed/upsert child chunks only.

3. **`src/rag_agent/nodes/retrieval.py`** — first real content here (ahead of the full Phase 3 graph, since it's the natural owner and Phase 3 imports it directly): after top-k child hits, map `parent_id` → dedupe → `load_parent()` → pass the larger parent text to the LLM instead of the tiny child snippet.

4. Schema changed → simplest path is deleting `qdrant_db/` contents and re-running ingestion rather than migrating in place (dev-only local storage).

**Verification**: re-run Phase 1 smoke questions, plus a question whose answer spans a full bio section — confirm the answer is now more complete than Phase 1's child-only version, with parent-level context shown.

---

## Phase 3 — Full LangGraph Agent

1. **`src/rag_agent/state.py`** first — `TypedDict`/pydantic state with `messages: Annotated[list, add_messages]` (LangGraph's built-in reducer), `rewritten_query`, `retrieved_chunks`, `summary`, `answer`.
2. **`src/rag_agent/prompts.py`** — string templates for query-rewrite, summarization, generation, and sub-question decomposition.
3. **`src/rag_agent/nodes/conversation.py`** — `rewrite_query(state)` resolves standalone queries across turns; `summarize_history(state)` condenses older turns into `summary` past a message-count threshold.
4. **`src/rag_agent/nodes/retrieval.py`** (upgrade) — start by wiring Phase 2's single-shot retrieval into the graph and get it working end-to-end; only then layer in map-reduce (decompose into sub-questions, fan out via LangGraph's `Send()`, reduce/dedupe results) as a follow-up within this phase, not simultaneously with graph wiring.
5. **`src/rag_agent/nodes/generation.py`** — `generate_answer(state)` from merged context + query + summary → LLM → `{"answer": ..., "messages": [AIMessage(...)]}`.
6. **`src/rag_agent/tools.py`** — retrieval wrapped as a LangChain `@tool` around `vector_store.search()` + parent lookup, for reuse/consistency even in a fixed-edge graph.
7. **`src/rag_agent/graph.py`** — `StateGraph(AgentState)`, nodes + edges (conditional edge to skip summarization when history is short), `.compile(checkpointer=...)`.
8. **`src/core/state_manager.py`** — **thin wrapper only; don't reinvent LangGraph's checkpointing.** `graph.compile(checkpointer=MemorySaver())` (fine for a single-process local Gradio app; resets on restart — acceptable for dev, swap to `SqliteSaver` later for persistence, a one-line change). `state_manager.py`'s actual job: own the checkpointer instance, generate/track a `thread_id` per Gradio session, expose `get_thread_config(thread_id)` for `graph.invoke()` calls. It should not store messages itself.
9. **`src/core/rag_system.py`** — refactor internals to `graph.invoke({"messages": [...]}, config=state_manager.get_thread_config(thread_id))`, keeping the same public `answer()` signature so `chat_interface.py` needs no changes.

**Verification**: multi-turn follow-up using a pronoun resolves correctly; restarting the app mid-conversation degrades gracefully (new thread, no crash); if map-reduce retrieval was implemented, a question spanning two bio sections shows both represented in the answer.

---

## Phase 4 — Hybrid Search + Additional Data Sources

1. **`src/ingestion/embeddings.py`** — add `fastembed.SparseTextEmbedding` (BM25) alongside the dense embedder.
2. **`src/ingestion/vector_store.py`** — switch to Qdrant named vectors (dense + sparse), hybrid query via `query_points(..., prefetch=[...], query=FusionQuery(fusion=Fusion.RRF))`. **Breaking schema change** — delete/recreate `qdrant_db/` and re-run ingestion.
3. **`src/web/github_connector.py`** — `PyGithub`-based fetch of READMEs/files from configured repos (`GITHUB_TOKEN`/`GITHUB_USERNAME`) → `Document`s tagged `source_type="github"`.
4. **`src/web/website_scraper.py`** — `requests` + `bs4` fetch/clean of configured URLs → `Document`s tagged `source_type="website"`.
5. **`src/ingestion/document_loader.py`** — extend `load_all_sources()` to call the new loaders, config-gated so behavior is unchanged when GitHub/URLs aren't configured.

**Verification**: ingest one real repo + one website; confirm a question answerable only from the new source retrieves correctly with accurate source attribution; compare an exact-keyword query before/after hybrid search is enabled.

---

## Phase 5 — Observability, Packaging, Test Hardening

1. **`src/core/execution_logger.py`** — Langfuse callback handler wrapping graph invocation, env-gated (no-op if `LANGFUSE_*` unset).
2. **`docker/docker-compose.yml` + `docker/Dockerfile`** — define Qdrant-as-server + app container, now that the app works. Watch the space-in-path gotcha for Windows bind mounts.
3. **Test hardening** — fill out `tests/test_ingestion.py`, `tests/test_retrieval.py`, `tests/test_agent.py` with real coverage, mocking Ollama/Qdrant so tests don't need a live model/process.

**Testing strategy across phases**: don't defer all testing to Phase 5, but don't front-load full coverage either. In Phases 1-3, rely on the manual smoke tests above for fast iteration, adding one or two lightweight pytest cases to the relevant test file as each module stabilizes (e.g., a basic chunk-count assertion once `simple_chunk` lands, a parent-lookup test once Phase 2's retrieval lands, a graph-compiles test once Phase 3's `graph.py` lands). Phase 5 is the dedicated hardening pass: edge cases, mocking external services, CI-runnable without live network/model dependencies.

---

## "Done" Criteria

**v1 (walking skeleton) = end of Phase 1**: `bio.md` ingested into a dense-only local Qdrant collection; simple top-k retrieval; single-turn Q&A via `ChatOllama` (plain function chain, no LangGraph); Gradio chat UI with grounded answers and visible source chunks. No GitHub/website ingestion, no Langfuse, no Docker, no formal tests — deferred by design.

**v2 "agentic" = Phases 2-5**: hierarchical parent-child chunking; hybrid dense+sparse retrieval; full LangGraph agent with query rewriting, summarization, map-reduce retrieval, and LangGraph-native checkpointing; GitHub + website ingestion; Langfuse observability; Docker packaging; real automated test suite.

---

## Critical Files (implementation entry points)

- `src/config.py` — LLM factory + settings, everything imports this first
- `src/ingestion/pipeline.py` — single ingestion seam, extended each phase
- `src/ingestion/vector_store.py` — Qdrant client, schema changes in Phases 2 & 4
- `src/rag_agent/graph.py` — LangGraph wiring, Phase 3
- `src/core/rag_system.py` — stable public interface (`answer()`) bridging UI to backend across all phases
