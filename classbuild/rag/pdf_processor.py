import hashlib
import io
import logging
import re
import unicodedata

from pypdf import PdfReader

from classbuild.rag.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_pages
from classbuild.rag.models import PageText, ProcessedDocument

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 10


class PDFProcessingError(Exception):
    """Base class; str(error) is a message written for users."""


class InvalidFileError(PDFProcessingError):
    pass


class FileTooLargeError(PDFProcessingError):
    pass


class EmptyFileError(PDFProcessingError):
    pass


class CorruptedPDFError(PDFProcessingError):
    pass


class EncryptedPDFError(PDFProcessingError):
    pass


class NoExtractableTextError(PDFProcessingError):
    pass


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pdf_upload(filename: str, data: bytes) -> None:
    if not filename or not filename.lower().endswith(".pdf"):
        raise InvalidFileError("Unsupported file type. Please upload a PDF file (.pdf).")
    if not data:
        raise EmptyFileError("The uploaded file is empty.")
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise FileTooLargeError(f"The file is larger than {MAX_FILE_SIZE_MB} MB. Please upload a smaller PDF.")
    if b"%PDF-" not in data[:1024]:
        raise InvalidFileError("This file is not a valid PDF (missing PDF header).")


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalize text while keeping paragraphs, code indentation and symbols."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    lines = []
    for line in text.split("\n"):
        stripped = line.lstrip(" \t")
        indent = line[: len(line) - len(stripped)]
        body = re.sub(r"[ \t]{2,}", " ", stripped).rstrip()
        lines.append(indent + body if body else "")
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n").rstrip()


def extract_pages(data: bytes) -> list[PageText]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:  # e.g. a missing crypto library
                raise EncryptedPDFError("This PDF is password-protected and cannot be read.") from exc
            if not decrypted:
                raise EncryptedPDFError("This PDF is password-protected and cannot be read.")
        total = len(reader.pages)
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise CorruptedPDFError("The PDF could not be read. It may be corrupted.") from exc

    if total == 0:
        raise CorruptedPDFError("The PDF has no pages.")

    pages = []
    for number in range(1, total + 1):
        try:
            raw = reader.pages[number - 1].extract_text() or ""
        except Exception:  # one bad page must not fail the document
            logger.warning("Could not extract text from page %s", number)
            raw = ""
        pages.append(PageText(page_number=number, text=clean_text(raw)))
    return pages


def process_pdf(
    filename: str,
    data: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ProcessedDocument:
    validate_pdf_upload(filename, data)
    pages = extract_pages(data)
    empty_pages = [page.page_number for page in pages if not page.text]
    if len(empty_pages) == len(pages):
        raise NoExtractableTextError(
            "No text could be extracted. The PDF may be scanned or image-only (OCR is not supported)."
        )
    chunks = chunk_pages(pages, filename, chunk_size, chunk_overlap)
    return ProcessedDocument(
        source_filename=filename,
        fingerprint=fingerprint(data),
        page_count=len(pages),
        text_page_count=len(pages) - len(empty_pages),
        empty_pages=empty_pages,
        word_count=sum(len(page.text.split()) for page in pages),
        char_count=sum(len(page.text) for page in pages),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=chunks,
    )
