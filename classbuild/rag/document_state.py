import logging
from dataclasses import dataclass

from classbuild.rag.models import ProcessedDocument
from classbuild.rag.pdf_processor import PDFProcessingError, fingerprint, process_pdf

logger = logging.getLogger(__name__)

DOC_KEY = "pdf_document"
DOC_STATUS_KEY = "pdf_status"
COURSE_MATERIAL_KEY = "course_material_fingerprint"
INDEX_KEY = "pdf_index"
SEARCH_KEY = "pdf_search"

GENERIC_ERROR = "Something went wrong while processing the PDF. Please try another file."


@dataclass(frozen=True)
class MaterialStatus:
    document: ProcessedDocument | None
    error: str | None
    processed_now: bool


def clear_learning_material(state) -> None:
    """Remove the document and everything derived from it."""
    for key in (DOC_KEY, DOC_STATUS_KEY, INDEX_KEY, SEARCH_KEY):
        state.pop(key, None)


def sync_learning_material(state, upload, processor=None) -> MaterialStatus:
    """`upload` is (filename, bytes) or None. A file is processed once per identity."""
    processor = processor or process_pdf
    if upload is None:
        clear_learning_material(state)
        return MaterialStatus(None, None, False)

    filename, data = upload
    identity = (filename, fingerprint(data))
    status = state.get(DOC_STATUS_KEY)
    if status is not None and status["identity"] == identity:
        return MaterialStatus(state.get(DOC_KEY), status["error"], False)

    clear_learning_material(state)  # a different file: drop everything from the old one
    document, error = None, None
    try:
        document = processor(filename, data)
    except PDFProcessingError as exc:
        error = str(exc)
    except Exception:
        logger.exception("Unexpected error while processing %s", filename)
        error = GENERIC_ERROR
    if document is not None:
        state[DOC_KEY] = document
    state[DOC_STATUS_KEY] = {"identity": identity, "error": error}
    return MaterialStatus(document, error, True)


def attach_material_to_course(state) -> None:
    """Record which PDF (if any) the course is generated with."""
    document = state.get(DOC_KEY)
    state[COURSE_MATERIAL_KEY] = document.fingerprint if document is not None else None


def material_for_course(state) -> ProcessedDocument | None:
    """The loaded PDF, only if it is the same file the course was generated with."""
    document = state.get(DOC_KEY)
    linked = state.get(COURSE_MATERIAL_KEY)
    if document is not None and linked is not None and document.fingerprint == linked:
        return document
    return None
