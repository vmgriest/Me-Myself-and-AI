# Hierarchical Parent-Child chunking
#
# Small "child" chunks get embedded and indexed in Qdrant for precise
# search, while larger "parent" chunks are saved to disk and fetched at
# query time to give the LLM richer context than a single small snippet
# ("small-to-big" retrieval).
import json  # to read/write the parent store as JSON

from langchain_core.documents import Document  # the type we split and return
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits text on paragraph/sentence/word boundaries

from src import config  # for chunk-size settings and PARENT_STORE_PATH


def parent_child_chunk(documents: list[Document]) -> list[Document]:
    """Splits each document into large parent chunks, then splits each
    parent into small child chunks. Parents are written to the parent
    store (data/parent_store/) keyed by parent_id; only the child chunks
    are returned, for embedding + indexing in Qdrant. Each child's
    metadata carries parent_id, pointing back to its parent's full text.
    """
    # two splitters: a coarse one for parents, a fine one for children within each parent
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE, chunk_overlap=config.PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE, chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )

    parents = parent_splitter.split_documents(documents)  # step 1: split into large section-sized chunks

    parent_records: dict[str, dict] = {}  # parent_id -> {"text": ..., "source": ...}, written to disk below
    child_chunks: list[Document] = []  # what we actually return (for embedding/indexing)

    for p_i, parent in enumerate(parents):  # walk through each parent chunk with its index
        source = parent.metadata.get("source", "unknown")  # e.g. "bio.md"
        parent_id = f"{source}-parent-{p_i}"  # stable id this parent can be looked up by later
        parent_records[parent_id] = {"text": parent.page_content, "source": source}  # queued for save_parents()

        children = child_splitter.split_documents([parent])  # step 2: split this one parent into smaller children
        for c_i, child in enumerate(children):  # walk through each child chunk with its index
            child.metadata["parent_id"] = parent_id  # the link back to the parent's full text
            child.metadata["chunk_id"] = f"{parent_id}-child-{c_i}"  # e.g. "bio.md-parent-0-child-0"
            child_chunks.append(child)

    save_parents(parent_records)  # persist parents to data/parent_store/parents.json
    return child_chunks


def save_parents(parent_records: dict[str, dict]) -> None:
    """Overwrites the parent store with this ingestion run's parent chunks."""
    config.PARENT_STORE_DIR.mkdir(parents=True, exist_ok=True)  # make sure the folder exists
    config.PARENT_STORE_PATH.write_text(json.dumps(parent_records, indent=2), encoding="utf-8")  # write as one JSON file


def load_parent(parent_id: str) -> str:
    """Reads a single parent chunk's full text back from the parent store,
    given the parent_id stored in a child chunk's metadata."""
    store = _load_parent_store()  # read (and cache) the whole parent store
    return store[parent_id]["text"]  # look up just this parent's text


_parent_store_cache: dict | None = None  # module-level cache so we don't re-read the file on every lookup


def _load_parent_store() -> dict:
    global _parent_store_cache
    if _parent_store_cache is None:  # only read from disk the first time
        _parent_store_cache = json.loads(config.PARENT_STORE_PATH.read_text(encoding="utf-8"))
    return _parent_store_cache
