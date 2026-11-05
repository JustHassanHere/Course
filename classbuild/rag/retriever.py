from dataclasses import dataclass

import numpy as np

from classbuild.rag.embeddings import EmbeddingService
from classbuild.rag.models import ProcessedDocument

DEFAULT_TOP_K = 3


class RetrievalError(Exception):
    """Base class; str(error) is a message written for users."""


class EmptyQueryError(RetrievalError):
    pass


class NoDocumentError(RetrievalError):
    pass


class IndexMismatchError(RetrievalError):
    pass


@dataclass(frozen=True, eq=False)
class DocumentIndex:
    document_fingerprint: str
    model_name: str
    chunk_ids: tuple[str, ...]
    embeddings: np.ndarray

    def __post_init__(self):
        self.embeddings.flags.writeable = False

    @property
    def size(self) -> int:
        return len(self.chunk_ids)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    chunk_id: str
    source_filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float


def cosine_similarity(query, matrix) -> np.ndarray:
    """cos(q, m) = (m . q) / (|m| |q|) for every row; zero vectors score 0."""
    q = np.asarray(query, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if q.ndim != 1 or m.ndim != 2 or m.shape[1] != q.shape[0]:
        raise ValueError("Vector shapes do not match.")
    denominator = np.linalg.norm(m, axis=1) * np.linalg.norm(q)
    dots = m @ q
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, dots / denominator, 0.0)
    return np.clip(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def build_index(document: ProcessedDocument, service: EmbeddingService) -> DocumentIndex:
    if document is None or not document.chunks:
        raise NoDocumentError("There is no PDF text to search.")
    matrix = service.embed_texts([chunk.text for chunk in document.chunks])
    return DocumentIndex(
        document_fingerprint=document.fingerprint,
        model_name=service.model_name,
        chunk_ids=tuple(chunk.chunk_id for chunk in document.chunks),
        embeddings=matrix,
    )


def check_index(index: DocumentIndex, document: ProcessedDocument) -> None:
    if index.document_fingerprint != document.fingerprint or index.chunk_ids != tuple(
        chunk.chunk_id for chunk in document.chunks
    ):
        raise IndexMismatchError("The search index belongs to a different PDF. Please search again.")


def search(
    index: DocumentIndex,
    document: ProcessedDocument,
    query: str,
    service: EmbeddingService,
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchResult]:
    if document is None:
        raise NoDocumentError("Please upload a PDF first.")
    if not query or not query.strip():
        raise EmptyQueryError("Please enter a question to search for.")
    check_index(index, document)
    scores = cosine_similarity(service.embed_query(query.strip()), index.embeddings)
    order = np.argsort(-scores, kind="stable")[:top_k]  # stable: ties keep document order
    results = []
    for rank, position in enumerate(order, start=1):
        chunk = document.chunks[int(position)]
        results.append(
            SearchResult(
                rank=rank,
                chunk_id=chunk.chunk_id,
                source_filename=chunk.source_filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=float(scores[int(position)]),
            )
        )
    return results
