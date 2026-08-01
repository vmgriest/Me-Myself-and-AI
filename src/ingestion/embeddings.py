# Dense (all-mpnet) & sparse (BM25) embeddings
#
# Phase 1 only implements the dense embedder. Sparse (BM25, via fastembed)
# gets added in Phase 4 for hybrid search.
from functools import lru_cache  # to cache the loaded model so it's only loaded once

from sentence_transformers import SentenceTransformer  # the embedding model wrapper

from src import config  # for config.DENSE_EMBED_MODEL


@lru_cache(maxsize=1)  # only ever holds one model instance per process
def _get_model() -> SentenceTransformer:
    # Cached so the ~420MB model only loads once per process, not per call.
    return SentenceTransformer(config.DENSE_EMBED_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = _get_model()  # reuse the cached model instance
    # normalize_embeddings=True is required for Qdrant's cosine-distance
    # search to behave correctly.
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)  # returns a numpy array
    return vectors.tolist()  # convert to plain Python lists so Qdrant/JSON can consume them


def embed_query(text: str) -> list[float]:
    return embed_documents([text])[0]  # embed a single string as a 1-item batch, then unwrap it
