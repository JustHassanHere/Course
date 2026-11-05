from dataclasses import dataclass

from classbuild.rag.document_state import DOC_KEY, INDEX_KEY, SEARCH_KEY
from classbuild.rag.embeddings import EmbeddingService
from classbuild.rag.retriever import (
    DEFAULT_TOP_K,
    DocumentIndex,
    EmptyQueryError,
    IndexMismatchError,
    NoDocumentError,
    SearchResult,
    build_index,
    check_index,
    search,
)


@dataclass(frozen=True)
class SearchOutcome:
    query: str
    results: list[SearchResult]
    index_built_now: bool


def has_current_index(state, document, service: EmbeddingService) -> bool:
    index: DocumentIndex | None = state.get(INDEX_KEY)
    if index is None or document is None or index.model_name != service.model_name:
        return False
    try:
        check_index(index, document)
    except IndexMismatchError:
        return False
    return True


def get_or_build_index(state, document, service: EmbeddingService) -> DocumentIndex:
    """Embed every chunk once per PDF; later calls reuse the stored index."""
    if not has_current_index(state, document, service):
        state[INDEX_KEY] = build_index(document, service)
    return state[INDEX_KEY]


def run_search(state, query: str, service: EmbeddingService | None = None, top_k: int = DEFAULT_TOP_K) -> SearchOutcome:
    state.pop(SEARCH_KEY, None)
    document = state.get(DOC_KEY)
    if document is None:
        raise NoDocumentError("Please upload a PDF first.")
    if not query or not query.strip():
        raise EmptyQueryError("Please enter a question to search for.")
    service = service or EmbeddingService()
    built_now = not has_current_index(state, document, service)
    index = get_or_build_index(state, document, service)
    results = search(index, document, query, service, top_k=top_k)
    state[SEARCH_KEY] = {"query": query.strip(), "results": results, "fingerprint": document.fingerprint}
    return SearchOutcome(query.strip(), results, built_now)


def current_results(state) -> SearchOutcome | None:
    """The last search, only if it belongs to the loaded PDF."""
    stored = state.get(SEARCH_KEY)
    document = state.get(DOC_KEY)
    if stored is None or document is None or stored["fingerprint"] != document.fingerprint:
        return None
    return SearchOutcome(stored["query"], stored["results"], False)
