from classbuild.rag.models import DocumentChunk, PageText

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

_SENTENCE_ENDS = (". ", "! ", "? ", ".\n", "!\n", "?\n")
_WEAK_SEPARATORS = ("\n", "; ", ", ", " ")


def _validate(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size.")


def _find_cut(text: str, start: int, end: int) -> int:
    """Choose where to end a chunk that would otherwise stop at `end`."""
    window = text[start:end]
    half = len(window) // 2
    for separators in (("\n\n",), _SENTENCE_ENDS, _WEAK_SEPARATORS):
        best = -1
        for separator in separators:
            position = window.rfind(separator)
            if position >= half:
                best = max(best, position + len(separator))
        if best > 0:
            return start + best
    return end


def _next_start(text: str, start: int, end: int, chunk_overlap: int) -> int:
    """Start the next chunk about `chunk_overlap` characters before `end`, at a word boundary."""
    candidate = max(end - chunk_overlap, start + 1)
    candidate = min(candidate, end)
    if 0 < candidate < len(text) and not text[candidate - 1].isspace() and not text[candidate].isspace():
        for position in range(candidate, end):
            if text[position].isspace():
                return min(position + 1, end)
    return candidate


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    _validate(chunk_size, chunk_overlap)
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _find_cut(text, start, end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = _next_start(text, start, end, chunk_overlap)
    return chunks


def chunk_pages(
    pages: list[PageText],
    source_filename: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Chunk every page separately; ids look like p3-c2."""
    chunks: list[DocumentChunk] = []
    for page in pages:
        for index, piece in enumerate(chunk_text(page.text, chunk_size, chunk_overlap), start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"p{page.page_number}-c{index}",
                    source_filename=source_filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=piece,
                )
            )
    return chunks
