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
RAW_DIR = DATA_DIR / "raw"  # unprocessed source material (bio.md, resume.pdf) — GitHub/websites are fetched live, not staged here
PARENT_STORE_DIR = DATA_DIR / "parent_store"  # parent chunks for hierarchical retrieval (Phase 2+)
PARENT_STORE_PATH = PARENT_STORE_DIR / "parents.json"  # parent_id -> parent text, written by ingestion
GITHUB_REPOS_MANIFEST_PATH = PARENT_STORE_DIR / "github_repos.json"  # complete repo name/description list, written by ingestion
BIO_PATH = RAW_DIR / "bio.md"  # gitignored — real content, see bio.example.md
RESUME_PATH = (BASE_DIR / os.getenv("RESUME_PATH", "./data/raw/resume.pdf")).resolve()  # gitignored, personal document

# --- Vector DB (Qdrant, embedded/local file mode — no server needed) ---
QDRANT_PATH = str((BASE_DIR / os.getenv("QDRANT_PATH", "./qdrant_db")).resolve())  # where Qdrant writes its local files
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "personal_ai")  # name of the collection storing chunk vectors

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # which branch get_llm() takes: ollama | groq | gemini | openai
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # where the local Ollama server listens
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # which locally-pulled model Ollama should run
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # cloud model served by Groq

# --- Embeddings & retrieval ---
DENSE_EMBED_MODEL = "all-mpnet-base-v2"  # sentence-transformers model, 768-dim output
RETRIEVAL_TOP_K = 4  # how many child chunks to retrieve per query

# --- Hierarchical parent-child chunking (Phase 2) ---
PARENT_CHUNK_SIZE = 800  # chars per parent chunk — larger, gives the LLM more surrounding context
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 250  # chars per child chunk — smaller, indexed in Qdrant for precise search
CHILD_CHUNK_OVERLAP = 50

# --- Data sources (Phase 4) ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional — unauthenticated GitHub API calls work but are rate-limited
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")  # whose public repos github_connector.py pulls READMEs from
_website_urls_raw = os.getenv("WEBSITE_URLS", "")  # comma-separated list of URLs for website_scraper.py to fetch
WEBSITE_URLS = [u.strip() for u in _website_urls_raw.split(",") if u.strip()]
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # used by website_scraper.py's Tavily-based fetcher


def get_llm(model_choice: str | None = None):
    """Returns a LangChain chat model for the given "provider:model" choice
    string (e.g. "groq:llama-3.3-70b-versatile", "ollama:llama3.2"). With no
    argument, falls back to LLM_PROVIDER/OLLAMA_MODEL/GROQ_MODEL from .env —
    this is what makes switching providers a config change, not a rewrite.

    Each non-default branch lazy-imports its package so you don't need
    langchain-google-genai/langchain-openai installed unless you actually
    use them.
    """
    if model_choice:  # e.g. from the UI's model dropdown
        provider, _, model = model_choice.partition(":")
    else:  # no explicit choice — use the .env defaults
        provider = LLM_PROVIDER
        model = {"ollama": OLLAMA_MODEL, "groq": GROQ_MODEL}.get(provider)

    if provider == "ollama":  # default, local dev path
        from langchain_ollama import ChatOllama  # imported here so cloud users don't need this package

        return ChatOllama(model=model, base_url=OLLAMA_BASE_URL)  # chat model pointed at local Ollama

    if provider == "groq":  # cloud option — fast inference, generous free tier
        api_key = os.getenv("GROQ_API_KEY")  # read the key from .env
        if not api_key:  # fail fast with a clear message instead of a confusing SDK error
            raise RuntimeError("provider=groq requires GROQ_API_KEY in .env")
        from langchain_groq import ChatGroq  # imported only when actually needed

        return ChatGroq(model=model, api_key=api_key)  # cloud chat model

    if provider == "gemini":  # cloud option
        api_key = os.getenv("GOOGLE_API_KEY")  # read the key from .env
        if not api_key:  # fail fast with a clear message instead of a confusing SDK error
            raise RuntimeError("provider=gemini requires GOOGLE_API_KEY in .env")
        from langchain_google_genai import ChatGoogleGenerativeAI  # imported only when actually needed

        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", google_api_key=api_key)  # cloud chat model

    if provider == "openai":  # cloud option
        api_key = os.getenv("OPENAI_API_KEY")  # read the key from .env
        if not api_key:  # same fail-fast pattern as the gemini branch
            raise RuntimeError("provider=openai requires OPENAI_API_KEY in .env")
        from langchain_openai import ChatOpenAI  # imported only when actually needed

        return ChatOpenAI(model=model or "gpt-4o-mini", api_key=api_key)  # cloud chat model

    raise ValueError(f"Unknown provider: {provider}")  # typo/misconfiguration guard


def default_model_choice() -> str:
    """The "provider:model" string matching .env's LLM_PROVIDER default —
    used to preselect the right option in the UI's model dropdown."""
    model = {"ollama": OLLAMA_MODEL, "groq": GROQ_MODEL}.get(LLM_PROVIDER, "")
    return f"{LLM_PROVIDER}:{model}"


def list_available_models() -> list[str]:
    """Returns "provider:model" choices that are actually usable right now:
    every Ollama model currently pulled locally (queried live, since that
    varies by machine), plus each cloud provider whose API key is set."""
    models: list[str] = []

    try:  # Ollama may not be running — don't let that break the whole list
        import requests

        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        response.raise_for_status()
        for m in response.json().get("models", []):
            models.append(f"ollama:{m['name']}")
    except Exception:
        pass  # Ollama unreachable — just don't offer its models

    if os.getenv("GROQ_API_KEY"):
        models.append(f"groq:{GROQ_MODEL}")
    if os.getenv("GOOGLE_API_KEY"):
        models.append("gemini:gemini-2.5-flash")
    if os.getenv("OPENAI_API_KEY"):
        models.append("openai:gpt-4o-mini")

    return models
