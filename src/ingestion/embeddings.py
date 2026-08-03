# Dense (all-mpnet) & sparse (BM25) embeddings
#
# Dense captures semantic similarity; sparse (BM25) captures exact keyword
# matches dense embeddings can miss (e.g. an exact project name). Phase 4's
# vector_store.py combines both via Qdrant's RRF fusion for hybrid search.
from functools import lru_cache  # to cache the loaded models so they're only loaded once

from fastembed import SparseTextEmbedding  # BM25-style sparse embedding model
from qdrant_client.models import SparseVector  # Qdrant's wire format for a sparse vector
from sentence_transformers import SentenceTransformer  # the dense embedding model wrapper

from src import config  # for config.DENSE_EMBED_MODEL

SPARSE_EMBED_MODEL = "Qdrant/bm25"  # fastembed's BM25 implementation, no training/corpus stats needed


@lru_cache(maxsize=1)  # only ever holds one model instance per process
def _get_dense_model() -> SentenceTransformer:
    # Cached so the ~420MB model only loads once per process, not per call.
    return SentenceTransformer(config.DENSE_EMBED_MODEL)


@lru_cache(maxsize=1)  # only ever holds one model instance per process
def _get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_EMBED_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = _get_dense_model()  # reuse the cached model instance
    # normalize_embeddings=True is required for Qdrant's cosine-distance
    # search to behave correctly.
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)  # returns a numpy array
    return vectors.tolist()  # convert to plain Python lists so Qdrant/JSON can consume them


def embed_query(text: str) -> list[float]:
    return embed_documents([text])[0]  # embed a single string as a 1-item batch, then unwrap it


def embed_sparse_documents(texts: list[str]) -> list[SparseVector]:
    model = _get_sparse_model()  # reuse the cached model instance
    embeddings = list(model.embed(texts))  # fastembed returns one SparseEmbedding per input text
    # convert fastembed's SparseEmbedding (numpy arrays) into Qdrant's SparseVector wire format
    return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embeddings]


def embed_sparse_query(text: str) -> SparseVector:
    return embed_sparse_documents([text])[0]  # embed a single string as a 1-item batch, then unwrap it
