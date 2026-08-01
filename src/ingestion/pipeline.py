# Runs ingestion end-to-end: load -> chunk -> embed -> index in Qdrant
#
# The single seam Phases 2 and 4 extend (parent-child chunking, hybrid
# embeddings, extra data sources) rather than rewrite. Run standalone
# with: python -m src.ingestion.pipeline
#
# Note: this must be run to completion (process exited) BEFORE launching
# src/main.py — Qdrant's local file mode only allows one open client.
from src import config  # for the QDRANT_PATH printed in the summary line
from src.ingestion import chunking, document_loader, embeddings, vector_store  # the four pipeline stages


def run_ingestion() -> None:
    documents = document_loader.load_all_sources()  # step 1: read raw sources into Documents
    chunks = chunking.simple_chunk(documents)  # step 2: split into smaller overlapping chunks

    texts = [chunk.page_content for chunk in chunks]  # pull out just the text for embedding
    vectors = embeddings.embed_documents(texts)  # step 3: turn each chunk's text into a vector

    client = vector_store.get_qdrant_client()  # open (or create) the local Qdrant store
    vector_store.ensure_collection(client)  # step 4a: make sure the collection exists
    vector_store.upsert_chunks(client, chunks, vectors)  # step 4b: write chunks + vectors into it

    print(f"Ingested {len(chunks)} chunks from {len(documents)} document(s) into {config.QDRANT_PATH}")  # summary log


if __name__ == "__main__":  # only runs when executed directly (python -m src.ingestion.pipeline)
    run_ingestion()
