# Qdrant client and collection management
#
# Uses Qdrant's embedded/local file mode (QdrantClient(path=...)) — no
# Docker or server process needed for dev. Phase 4: the collection now
# holds TWO named vectors per point ("dense" + "sparse"), combined at
# query time via Reciprocal Rank Fusion (RRF) for hybrid search — dense
# catches semantic matches, sparse (BM25) catches exact keyword matches
# dense embeddings sometimes miss.
from functools import lru_cache  # to cache the single client instance

from langchain_core.documents import Document  # type hint for the chunks we upsert
from qdrant_client import QdrantClient  # the Qdrant SDK's client class
from qdrant_client.models import (  # types for collection/point config and hybrid queries
    Distance,
    FusionQuery,
    Prefetch,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import Fusion

from src import config  # for QDRANT_PATH, QDRANT_COLLECTION, RETRIEVAL_TOP_K

DENSE_VECTOR_SIZE = 768  # all-mpnet-base-v2 output dimension
DENSE_VECTOR_NAME = "dense"  # name of the dense vector field within each point
SPARSE_VECTOR_NAME = "sparse"  # name of the sparse (BM25) vector field within each point


@lru_cache(maxsize=1)  # ensures only one QdrantClient is ever created in this process
def get_qdrant_client() -> QdrantClient:
    # Local file mode holds an exclusive lock on QDRANT_PATH, so only one
    # client may be open per process — cached here to enforce that.
    return QdrantClient(path=config.QDRANT_PATH)  # opens/creates the local Qdrant storage directory


def ensure_collection(client: QdrantClient) -> None:
    """Creates the collection (with both named vectors) if it doesn't
    already exist. Safe to call every ingestion run."""
    if not client.collection_exists(config.QDRANT_COLLECTION):  # skip creation if it's already there
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config={DENSE_VECTOR_NAME: VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )


def upsert_chunks(
    client: QdrantClient,
    chunks: list[Document],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
) -> None:
    """Writes chunks + their dense and sparse vectors into Qdrant. Point
    ids are just the list index, so re-running ingestion overwrites the
    previous points (idempotent for a given chunk count)."""
    points = [
        PointStruct(
            id=i,  # sequential id — re-ingestion overwrites points with the same index
            vector={DENSE_VECTOR_NAME: dense_vec, SPARSE_VECTOR_NAME: sparse_vec},  # both named vectors for this chunk
            payload={  # extra data stored alongside the vectors, returned on search
                "text": chunk.page_content,  # the child chunk's text (only used as a search key, not shown to the LLM)
                "source": chunk.metadata.get("source"),  # e.g. "bio.md", shown in the UI as attribution
                "source_type": chunk.metadata.get("source_type"),  # e.g. "bio", "github", "website"
                "chunk_id": chunk.metadata.get("chunk_id"),  # for debugging/tracing
                "parent_id": chunk.metadata.get("parent_id"),  # links back to the full parent text in parent_store/
            },
        )
        for i, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors))  # pair each chunk with both its embeddings
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)  # write all points in one call


def search(
    client: QdrantClient,
    dense_query_vector: list[float],
    sparse_query_vector: SparseVector,
    k: int = config.RETRIEVAL_TOP_K,
):
    """Hybrid top-k search: runs dense and sparse search in parallel
    (prefetch), then fuses the two ranked lists with Reciprocal Rank
    Fusion. Returns Qdrant ScoredPoint objects (.payload, .score, .id)."""
    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            Prefetch(query=dense_query_vector, using=DENSE_VECTOR_NAME, limit=k * 2),  # semantic candidates
            Prefetch(query=sparse_query_vector, using=SPARSE_VECTOR_NAME, limit=k * 2),  # exact-keyword candidates
        ],
        query=FusionQuery(fusion=Fusion.RRF),  # merges both ranked lists into one
        limit=k,
    )
    return response.points  # list of ScoredPoint, best match first
