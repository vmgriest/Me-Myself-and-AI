# Hierarchical Parent-Child chunking
#
# Phase 1 only has simple_chunk() — flat, fixed-size splitting. Phase 2 adds
# parent_child_chunk() alongside it (small "child" chunks for search, larger
# "parent" chunks for context) without changing this function's interface.
from langchain_core.documents import Document  # the type we split and return
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits text on paragraph/sentence/word boundaries


def simple_chunk(documents: list[Document], chunk_size: int = 600, chunk_overlap: int = 100) -> list[Document]:
    """Splits documents into overlapping fixed-size chunks and tags each
    with a stable chunk_id (used for debugging/tracing which chunk a
    retrieved answer came from)."""
    # chunk_size/overlap in characters; overlap avoids cutting a sentence in half at a boundary
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)  # returns a new list of smaller Documents, metadata preserved
    for i, chunk in enumerate(chunks):  # walk through each chunk with its index
        source = chunk.metadata.get("source", "unknown")  # pull the original filename from metadata
        chunk.metadata["chunk_id"] = f"{source}-{i}"  # e.g. "bio.md-0", "bio.md-1", ...
    return chunks
