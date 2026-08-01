# Central configuration (models, paths, settings)
#
# Every other module imports this file first for paths/settings, so it has
# no dependency on anything else in src/.
from pathlib import Path  # for filesystem paths that work cross-platform

from dotenv import load_dotenv  # reads key=value pairs from a .env file into the environment
import os  # to read environment variables via os.getenv

# Project root = two levels up from this file (src/config.py -> src/ -> root)
BASE_DIR = Path(__file__).resolve().parent.parent  # absolute path to the repo root
load_dotenv(BASE_DIR / ".env")  # loads .env values into os.environ before we read them below

# --- Data locations ---
DATA_DIR = BASE_DIR / "data"  # root of all data files
RAW_DIR = DATA_DIR / "raw"  # unprocessed source material (bio.md, github/, websites/)
PROCESSED_DIR = DATA_DIR / "processed"  # chunked/cleaned docs after ingestion (Phase 2+)
PARENT_STORE_DIR = DATA_DIR / "parent_store"  # parent chunks for hierarchical retrieval (Phase 2+)
BIO_PATH = RAW_DIR / "bio.md"  # gitignored — real content, see bio.example.md

# --- Vector DB (Qdrant, embedded/local file mode — no server needed) ---
QDRANT_PATH = str((BASE_DIR / os.getenv("QDRANT_PATH", "./qdrant_db")).resolve())  # where Qdrant writes its local files
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "personal_ai")  # name of the collection storing chunk vectors

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # which branch get_llm() takes: ollama | gemini | openai
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # where the local Ollama server listens
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # which locally-pulled model Ollama should run

# --- Embeddings & retrieval ---
DENSE_EMBED_MODEL = "all-mpnet-base-v2"  # sentence-transformers model, 768-dim output
RETRIEVAL_TOP_K = 4  # how many chunks to retrieve per query


def get_llm():
    """Returns a LangChain chat model, chosen via LLM_PROVIDER so switching
    providers later is a config change, not a rewrite.

    Each non-default branch lazy-imports its package so you don't need
    langchain-google-genai/langchain-openai installed unless you actually
    use them.
    """
    if LLM_PROVIDER == "ollama":  # default, local dev path
        from langchain_ollama import ChatOllama  # imported here so cloud users don't need this package

        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)  # chat model pointed at local Ollama

    if LLM_PROVIDER == "gemini":  # cloud option
        api_key = os.getenv("GOOGLE_API_KEY")  # read the key from .env
        if not api_key:  # fail fast with a clear message instead of a confusing SDK error
            raise RuntimeError("LLM_PROVIDER=gemini requires GOOGLE_API_KEY in .env")
        from langchain_google_genai import ChatGoogleGenerativeAI  # imported only when actually needed

        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)  # cloud chat model

    if LLM_PROVIDER == "openai":  # cloud option
        api_key = os.getenv("OPENAI_API_KEY")  # read the key from .env
        if not api_key:  # same fail-fast pattern as the gemini branch
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY in .env")
        from langchain_openai import ChatOpenAI  # imported only when actually needed

        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key)  # cloud chat model

    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")  # typo/misconfiguration guard
