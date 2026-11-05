import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEVICE = "cpu"
BATCH_SIZE = 32


class EmbeddingError(Exception):
    """Base class; str(error) is a message written for users."""


class EmbeddingInputError(EmbeddingError):
    pass


class EmbeddingModelUnavailableError(EmbeddingError):
    pass


class EmbeddingResourceError(EmbeddingError):
    pass


class InvalidEmbeddingError(EmbeddingError):
    pass


class TextEncoder(Protocol):
    def encode(self, texts, **kwargs): ...


@lru_cache(maxsize=2)
def load_model(model_name: str = MODEL_NAME, device: str = DEVICE):
    """Load the model once per process. The first use downloads it (about 90 MB)."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)
    except MemoryError as exc:
        raise EmbeddingResourceError("Not enough memory to load the embedding model.") from exc
    except Exception as exc:
        logger.warning("Embedding model could not be loaded: %s", exc)
        raise EmbeddingModelUnavailableError(
            "The local embedding model could not be loaded. Check your internet connection "
            "for the first download and that sentence-transformers is installed."
        ) from exc
    if model.get_sentence_embedding_dimension() != EMBEDDING_DIM:
        raise EmbeddingModelUnavailableError("The embedding model has an unexpected dimension.")
    return model


def validate_embedding_matrix(matrix, expected_rows: int | None = None, dimension: int = EMBEDDING_DIM):
    try:
        array = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidEmbeddingError("The embedding model returned invalid vectors.") from exc
    if array.ndim != 2 or array.shape[1] != dimension:
        raise InvalidEmbeddingError("The embedding model returned vectors of the wrong shape.")
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise InvalidEmbeddingError("The embedding model returned the wrong number of vectors.")
    if not np.all(np.isfinite(array)):
        raise InvalidEmbeddingError("The embedding model returned invalid numbers.")
    if np.any(np.linalg.norm(array, axis=1) == 0):
        raise InvalidEmbeddingError("The embedding model returned an empty vector.")
    return array


class EmbeddingService:
    def __init__(self, model: TextEncoder | None = None, model_name: str = MODEL_NAME):
        self._model = model
        self.model_name = model_name

    def _get_model(self):
        return self._model if self._model is not None else load_model(self.model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingInputError("Cannot embed empty text.")
        model = self._get_model()
        try:
            matrix = model.encode(
                list(texts), batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True
            )
        except MemoryError as exc:
            raise EmbeddingResourceError("Not enough memory to compute embeddings.") from exc
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            raise EmbeddingModelUnavailableError("The embedding model failed to process the text.") from exc
        return validate_embedding_matrix(matrix, expected_rows=len(texts))

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
