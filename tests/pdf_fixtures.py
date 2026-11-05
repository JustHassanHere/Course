"""Build small PDFs in memory. No binary fixture files are needed."""

import io

from pypdf import PdfReader, PdfWriter


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages) -> bytes:
    """`pages` holds one string per text page, or None for an image-only page."""
    count = len(pages)
    page_ids = [4 + 2 * i for i in range(count)]
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {count} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    for position, text in enumerate(pages):
        page_id, content_id = 4 + 2 * position, 5 + 2 * position
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode()
        if text is None:
            stream = b"0.8 g 100 100 200 200 re f"  # a grey box instead of text
        else:
            lines = ["BT /F1 11 Tf 72 720 Td 14 TL"]
            lines += [f"({_escape(line)}) Tj T*" for line in text.split("\n")]
            lines.append("ET")
            stream = "\n".join(lines).encode("latin-1")
        objects[content_id] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    output = b"%PDF-1.4\n"
    offsets = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(output)
        output += f"{object_id} 0 obj\n".encode() + objects[object_id] + b"\nendobj\n"
    xref_at = len(output)
    size = max(objects) + 1
    output += f"xref\n0 {size}\n".encode() + b"0000000000 65535 f \n"
    for object_id in range(1, size):
        output += f"{offsets[object_id]:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return output


def make_encrypted_pdf(pages, user_password="secret") -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf(pages))))
    writer.encrypt(user_password, algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
