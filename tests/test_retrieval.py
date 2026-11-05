import unittest

import numpy as np

from classbuild.rag import (
    EMBEDDING_DIM,
    DOC_KEY,
    INDEX_KEY,
    SEARCH_KEY,
    EmbeddingInputError,
    EmbeddingModelUnavailableError,
    EmbeddingResourceError,
    EmbeddingService,
    EmptyQueryError,
    IndexMismatchError,
    InvalidEmbeddingError,
    NoDocumentError,
    build_index,
    cosine_similarity,
    current_results,
    get_or_build_index,
    process_pdf,
    run_search,
    search,
    sync_learning_material,
    validate_embedding_matrix,
)
from tests.embedding_fakes import BrokenEncoder, FakeEncoder
from tests.groq_fakes import PYTHON_PAGES
from tests.pdf_fixtures import make_pdf


def python_document():
    return process_pdf("python.pdf", make_pdf(PYTHON_PAGES))


class EmbeddingServiceTests(unittest.TestCase):
    def test_output_shape_and_batching(self):
        encoder = FakeEncoder()
        matrix = EmbeddingService(model=encoder).embed_texts(["python list", "python tuple", "loop"])
        self.assertEqual(matrix.shape, (3, EMBEDDING_DIM))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(len(encoder.calls), 1)  # one batched call

    def test_query_vector(self):
        vector = EmbeddingService(model=FakeEncoder()).embed_query("python list")
        self.assertEqual(vector.shape, (EMBEDDING_DIM,))

    def test_rejects_bad_input(self):
        service = EmbeddingService(model=FakeEncoder())
        for bad in ([], [""], ["ok", "   "], [None]):
            with self.assertRaises(EmbeddingInputError):
                service.embed_texts(bad)

    def test_model_failures_become_readable_errors(self):
        with self.assertRaises(EmbeddingModelUnavailableError):
            EmbeddingService(model=BrokenEncoder(error=OSError("no network"))).embed_texts(["x"])
        with self.assertRaises(EmbeddingResourceError):
            EmbeddingService(model=BrokenEncoder(error=MemoryError())).embed_texts(["x"])

    def test_invalid_vectors_are_rejected(self):
        cases = [
            np.zeros((1, EMBEDDING_DIM)),  # zero vector
            np.full((1, EMBEDDING_DIM), np.nan),
            np.ones((1, 10)),  # wrong dimension
            np.ones((2, EMBEDDING_DIM)),  # wrong row count
        ]
        for output in cases:
            with self.assertRaises(InvalidEmbeddingError):
                EmbeddingService(model=BrokenEncoder(output=output)).embed_texts(["x"])
        with self.assertRaises(InvalidEmbeddingError):
            validate_embedding_matrix([[1, 2, 3]], expected_rows=1)


class CosineTests(unittest.TestCase):
    def test_known_values(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [2.0, 2.0]])
        scores = cosine_similarity(np.array([1.0, 0.0]), matrix)
        np.testing.assert_allclose(scores, [1.0, 0.0, -1.0, 2**-0.5], atol=1e-9)

    def test_matches_manual_formula(self):
        rng = np.random.default_rng(1)
        q, m = rng.normal(size=8), rng.normal(size=(5, 8))
        manual = [float(row @ q / (np.linalg.norm(row) * np.linalg.norm(q))) for row in m]
        np.testing.assert_allclose(cosine_similarity(q, m), manual, atol=1e-9)

    def test_zero_vectors_score_zero(self):
        scores = cosine_similarity(np.array([1.0, 1.0]), np.array([[0.0, 0.0], [1.0, 1.0]]))
        np.testing.assert_allclose(scores, [0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(cosine_similarity(np.zeros(2), np.ones((2, 2))), [0.0, 0.0])

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity(np.ones(3), np.ones((2, 4)))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.document = python_document()
        self.service = EmbeddingService(model=FakeEncoder())
        self.index = build_index(self.document, self.service)

    def test_top_k_ordering_and_real_scores(self):
        results = search(self.index, self.document, "lists tuples mutable immutable", self.service)
        self.assertEqual(results[0].page_number, 1)
        self.assertEqual([r.rank for r in results], [1, 2])
        self.assertGreater(results[0].score, results[1].score)
        self.assertGreater(results[0].score, 0.3)

    def test_second_topic_finds_page_two(self):
        results = search(self.index, self.document, "function loop indentation", self.service)
        self.assertEqual(results[0].page_number, 2)

    def test_metadata_is_unchanged(self):
        result = search(self.index, self.document, "dynamically typed variable", self.service)[0]
        chunk = self.document.get_chunk(result.chunk_id)
        self.assertEqual((result.text, result.page_number, result.source_filename),
                         (chunk.text, chunk.page_number, chunk.source_filename))

    def test_unrelated_question_scores_low(self):
        results = search(self.index, self.document, "How do Kubernetes pods work?", self.service)
        self.assertTrue(all(r.score < 0.30 for r in results))

    def test_ties_keep_document_order(self):
        results = search(self.index, self.document, "something with no keywords", self.service)
        self.assertEqual([r.chunk_id for r in results], [c.chunk_id for c in self.document.chunks])

    def test_empty_query_and_no_document(self):
        with self.assertRaises(EmptyQueryError):
            search(self.index, self.document, "   ", self.service)
        with self.assertRaises(NoDocumentError):
            search(self.index, None, "python", self.service)

    def test_refuses_index_from_another_document(self):
        other = process_pdf("o.pdf", make_pdf(["python lists are great"]))
        with self.assertRaises(IndexMismatchError):
            search(self.index, other, "python", self.service)


class SearchStateTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        sync_learning_material(self.state, ("python.pdf", make_pdf(PYTHON_PAGES)))
        self.encoder = FakeEncoder()
        self.service = EmbeddingService(model=self.encoder)

    def test_embeddings_are_computed_once_per_pdf(self):
        first = run_search(self.state, "python lists", self.service)
        second = run_search(self.state, "python function", self.service)
        self.assertTrue(first.index_built_now)
        self.assertFalse(second.index_built_now)
        chunk_batches = [c for c in self.encoder.calls if len(c) > 1]
        self.assertEqual(len(chunk_batches), 1)
        self.assertEqual(len(self.encoder.calls), 3)  # 1 index build + 2 queries

    def test_replacing_pdf_invalidates_index_and_results(self):
        run_search(self.state, "python lists", self.service)
        self.assertIsNotNone(current_results(self.state))
        sync_learning_material(self.state, ("other.pdf", make_pdf(["python strings"])))
        self.assertNotIn(INDEX_KEY, self.state)
        self.assertIsNone(current_results(self.state))

    def test_removing_pdf_clears_everything(self):
        run_search(self.state, "python lists", self.service)
        sync_learning_material(self.state, None)
        for key in (DOC_KEY, INDEX_KEY, SEARCH_KEY):
            self.assertNotIn(key, self.state)

    def test_stale_results_are_hidden(self):
        run_search(self.state, "python lists", self.service)
        self.state[DOC_KEY] = process_pdf("x.pdf", make_pdf(["python strings"]))
        self.assertIsNone(current_results(self.state))

    def test_errors_before_any_embedding_work(self):
        with self.assertRaises(EmptyQueryError):
            run_search(self.state, " ", self.service)
        self.assertEqual(self.encoder.calls, [])
        with self.assertRaises(NoDocumentError):
            run_search({}, "python", self.service)

    def test_get_or_build_index_reuses_stored_index(self):
        document = self.state[DOC_KEY]
        first = get_or_build_index(self.state, document, self.service)
        second = get_or_build_index(self.state, document, self.service)
        self.assertIs(first, second)
        self.assertFalse(first.embeddings.flags.writeable)


if __name__ == "__main__":
    unittest.main()
