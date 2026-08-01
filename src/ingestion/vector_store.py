# Qdrant client and collection management
#
# Uses Qdrant's embedded/local file mode (QdrantClient(path=...)) — no
# Docker or server process needed for dev.
from functools import lru_cache  # to cache the single client instance

from langchain_core.documents import Document  # type hint for the chunks we upsert
from qdrant_client import QdrantClient  # the Qdrant SDK's client class
from qdrant_client.models import Distance, PointStruct, VectorParams  # types for collection/point config

from src import config  # for QDRANT_PATH, QDRANT_COLLECTION, RETRIEVAL_TOP_K

DENSE_VECTOR_SIZE = 768  # all-mpnet-base-v2 output dimension


@lru_cache(maxsize=1)  # ensures only one QdrantClient is ever created in this process
def get_qdrant_client() -> QdrantClient:
    # Local file mode holds an exclusive lock on QDRANT_PATH, so only one
    # client may be open per process — cached here to enforce that.
    return QdrantClient(path=config.QDRANT_PATH)  # opens/creates the local Qdrant storage directory


def ensure_collection(client: QdrantClient) -> None:
    """Creates the collection if it doesn't already exist. Safe to call
    every ingestion run."""
    if not client.collection_exists(config.QDRANT_COLLECTION):  # skip creation if it's already there
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE),  # 768-dim, cosine similarity
        )


def upsert_chunks(client: QdrantClient, chunks: list[Document], embeddings: list[list[float]]) -> None:
    """Writes chunks + their vectors into Qdrant. Point ids are just the
    list index, so re-running ingestion overwrites the previous points
    (idempotent for a given chunk count)."""
    points = [
        PointStruct(
            id=i,  # sequential id — re-ingestion overwrites points with the same index
            vector=vector,  # the embedding for this chunk's text
            payload={  # extra data stored alongside the vector, returned on search
                "text": chunk.page_content,  # the actual chunk text, used to build the LLM prompt
                "source": chunk.metadata.get("source"),  # e.g. "bio.md", shown in the UI as attribution
                "source_type": chunk.metadata.get("source_type"),  # e.g. "bio", "github", "website"
                "chunk_id": chunk.metadata.get("chunk_id"),  # for debugging/tracing
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings))  # pair each chunk with its embedding
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)  # write all points in one call


def search(client: QdrantClient, query_vector: list[float], k: int = config.RETRIEVAL_TOP_K):
    """Top-k similarity search. Returns Qdrant ScoredPoint objects
    (.payload, .score, .id) — not yet plain dicts, since rag_system.py
    reads .payload/.score directly."""
    response = client.query_points(collection_name=config.QDRANT_COLLECTION, query=query_vector, limit=k)  # runs the similarity search
    return response.points  # list of ScoredPoint, best match first
