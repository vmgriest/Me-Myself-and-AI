# Loads data from multiple sources (PDF, MD, Web)
#
# Phase 1 only reads bio.md. load_all_sources() is the single seam later
# phases extend (GitHub repos, scraped websites) without callers changing.
from langchain_core.documents import Document  # LangChain's standard text + metadata container

from src import config  # for config.BIO_PATH


def load_bio() -> list[Document]:
    text = config.BIO_PATH.read_text(encoding="utf-8")  # read the whole bio file as one string
    # wrap it as a single Document; metadata lets downstream code attribute chunks back to "bio.md"
    return [Document(page_content=text, metadata={"source": "bio.md", "source_type": "bio"})]


def load_all_sources() -> list[Document]:
    return load_bio()  # only source in Phase 1; Phase 4 will also call github/website loaders here
