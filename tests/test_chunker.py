import unittest

from classbuild.rag import PageText, chunk_pages, chunk_text


def long_text(paragraphs=6, sentences=8):
    return "\n\n".join(
        " ".join(f"Sentence {p}-{s} talks about Python lists and tuples." for s in range(sentences))
        for p in range(paragraphs)
    )


class ChunkTextTests(unittest.TestCase):
    def test_size_limit(self):
        for chunk in chunk_text(long_text()):
            self.assertLessEqual(len(chunk), 800)

    def test_empty_and_short_text(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n "), [])
        self.assertEqual(chunk_text("short text"), ["short text"])

    def test_consecutive_chunks_overlap(self):
        chunks = chunk_text(long_text())
        self.assertGreater(len(chunks), 2)
        for first, second in zip(chunks, chunks[1:]):
            overlap = [n for n in range(1, 400) if first.endswith(second[:n])]
            self.assertTrue(overlap, "consecutive chunks should share some text")

    def test_full_ordered_coverage(self):
        text = long_text()
        chunks = chunk_text(text)
        position = 0
        for chunk in chunks:
            found = text.find(chunk, max(0, position - 400))
            self.assertNotEqual(found, -1)
            self.assertLessEqual(found, position)  # no gap before this chunk
            position = max(position, found + len(chunk))
        self.assertEqual(position, len(text.strip()))

    def test_prefers_paragraph_boundaries(self):
        text = ("a" * 100 + " ") * 4 + "\n\n" + ("b" * 100 + " ") * 6
        first = chunk_text(text)[0]
        self.assertTrue(first.endswith("a"), "the cut should fall at the paragraph break")

    def test_prefers_sentence_boundaries(self):
        text = "Word " * 150 + "ends here. " + "More words follow " * 40
        first = chunk_text(text)[0]
        self.assertTrue(first.endswith("."))

    def test_one_very_long_word(self):
        chunks = chunk_text("x" * 2000)
        self.assertTrue(all(len(c) <= 800 for c in chunks))
        self.assertGreaterEqual(len(chunks), 3)

    def test_no_duplicate_tail_chunk(self):
        chunks = chunk_text(long_text())
        self.assertEqual(len(chunks), len(set(chunks)))
        self.assertNotIn(chunks[-1], chunks[-2])

    def test_zero_overlap(self):
        chunks = chunk_text("word " * 500, chunk_size=200, chunk_overlap=0)
        self.assertEqual(sum(len(c.split()) for c in chunks), 500)

    def test_invalid_settings(self):
        for size, overlap in [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)]:
            with self.assertRaises(ValueError):
                chunk_text("text", size, overlap)


class ChunkPagesTests(unittest.TestCase):
    def test_ids_and_metadata(self):
        pages = [PageText(page_number=1, text=long_text(4)), PageText(page_number=3, text="Short page.")]
        chunks = chunk_pages(pages, "doc.pdf")
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("p3-c1", ids)
        self.assertEqual(chunks[0].chunk_id, "p1-c1")
        self.assertEqual(chunks[1].chunk_id, "p1-c2")
        for chunk in chunks:
            self.assertEqual(chunk.source_filename, "doc.pdf")
            self.assertEqual(chunk.chunk_id, f"p{chunk.page_number}-c{chunk.chunk_index}")

    def test_chunks_never_span_pages(self):
        pages = [PageText(page_number=1, text="alpha text"), PageText(page_number=2, text="beta text")]
        texts = {c.page_number: c.text for c in chunk_pages(pages, "d.pdf")}
        self.assertEqual(texts, {1: "alpha text", 2: "beta text"})

    def test_empty_pages_produce_no_chunks(self):
        pages = [PageText(page_number=1, text=""), PageText(page_number=2, text="content")]
        self.assertEqual([c.chunk_id for c in chunk_pages(pages, "d.pdf")], ["p2-c1"])


if __name__ == "__main__":
    unittest.main()
