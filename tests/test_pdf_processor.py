import unittest
from unittest.mock import patch

from classbuild.rag import (
    CorruptedPDFError,
    EmptyFileError,
    EncryptedPDFError,
    FileTooLargeError,
    InvalidFileError,
    NoExtractableTextError,
    clean_text,
    extract_pages,
    fingerprint,
    process_pdf,
    validate_pdf_upload,
)
from tests.groq_fakes import PYTHON_PAGES
from tests.pdf_fixtures import make_encrypted_pdf, make_pdf


class ValidationTests(unittest.TestCase):
    def test_rejects_wrong_extension(self):
        with self.assertRaises(InvalidFileError):
            validate_pdf_upload("notes.txt", b"%PDF-1.4 data")

    def test_extension_is_case_insensitive(self):
        validate_pdf_upload("NOTES.PDF", b"%PDF-1.4 data")

    def test_rejects_empty_file(self):
        with self.assertRaises(EmptyFileError):
            validate_pdf_upload("a.pdf", b"")

    def test_rejects_oversized_file(self):
        with self.assertRaises(FileTooLargeError):
            validate_pdf_upload("a.pdf", b"%PDF-" + b"x" * (10 * 1024 * 1024))

    def test_rejects_non_pdf_content(self):
        with self.assertRaises(InvalidFileError):
            validate_pdf_upload("a.pdf", b"just plain text, not a pdf")


class ExtractionTests(unittest.TestCase):
    def test_multi_page_text(self):
        pages = extract_pages(make_pdf(PYTHON_PAGES))
        self.assertEqual([p.page_number for p in pages], [1, 2])
        self.assertIn("dynamically typed", pages[0].text)
        self.assertIn("function", pages[1].text)

    def test_mixed_image_and_text(self):
        document = process_pdf("m.pdf", make_pdf(["Some text here", None, "More text"]))
        self.assertEqual(document.empty_pages, [2])
        self.assertEqual(document.text_page_count, 2)
        self.assertEqual({c.page_number for c in document.chunks}, {1, 3})

    def test_scanned_pdf_has_no_text(self):
        with self.assertRaises(NoExtractableTextError):
            process_pdf("s.pdf", make_pdf([None, None]))

    def test_corrupted_pdf(self):
        with self.assertRaises(CorruptedPDFError):
            process_pdf("c.pdf", b"%PDF-1.4\nthis is not really a pdf at all")

    def test_encrypted_pdf(self):
        with self.assertRaises(EncryptedPDFError):
            process_pdf("e.pdf", make_encrypted_pdf(PYTHON_PAGES))

    def test_one_failing_page_is_skipped(self):
        data = make_pdf(["First page text", "Second page text"])
        from pypdf import PageObject

        original = PageObject.extract_text
        calls = {"n": 0}

        def flaky(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return original(self, *args, **kwargs)

        with patch.object(PageObject, "extract_text", flaky):
            document = process_pdf("f.pdf", data)
        self.assertEqual(document.empty_pages, [1])
        self.assertEqual(document.text_page_count, 1)


class ProcessPdfTests(unittest.TestCase):
    def test_statistics_and_fingerprint(self):
        data = make_pdf(PYTHON_PAGES)
        document = process_pdf("python.pdf", data)
        self.assertEqual(document.fingerprint, fingerprint(data))
        self.assertEqual(document.page_count, 2)
        self.assertEqual(document.chunk_count, len(document.chunks))
        self.assertGreater(document.word_count, 40)
        self.assertEqual(document.chunk_size, 800)
        self.assertEqual(document.chunk_overlap, 150)
        self.assertEqual(len({c.chunk_id for c in document.chunks}), document.chunk_count)

    def test_different_files_have_different_fingerprints(self):
        self.assertNotEqual(fingerprint(make_pdf(["a"])), fingerprint(make_pdf(["b"])))

    def test_no_groq_call_and_nothing_written_to_disk(self):
        with patch("builtins.open", side_effect=AssertionError("disk access")):
            with patch("classbuild.llm.groq_client.Groq", side_effect=AssertionError("groq call")):
                process_pdf("python.pdf", make_pdf(PYTHON_PAGES))


class CleanTextTests(unittest.TestCase):
    def test_normalizes_whitespace_but_keeps_paragraphs(self):
        cleaned = clean_text("one   two\r\n\r\n\r\n\r\nthree\t\tfour  ")
        self.assertEqual(cleaned, "one two\n\nthree four")

    def test_keeps_code_indentation_and_symbols(self):
        cleaned = clean_text("def f(x):\n    return x * 2 # double\n")
        self.assertEqual(cleaned, "def f(x):\n    return x * 2 # double")

    def test_removes_control_characters(self):
        self.assertEqual(clean_text("a\x00b\x07c"), "abc")


if __name__ == "__main__":
    unittest.main()
