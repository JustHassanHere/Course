import unittest

from classbuild.rag import (
    COURSE_MATERIAL_KEY,
    DOC_KEY,
    DOC_STATUS_KEY,
    INDEX_KEY,
    SEARCH_KEY,
    attach_material_to_course,
    clear_learning_material,
    material_for_course,
    process_pdf,
    sync_learning_material,
)
from classbuild.rag.document_state import GENERIC_ERROR
from tests.groq_fakes import PYTHON_PAGES
from tests.pdf_fixtures import make_pdf


class Counting:
    def __init__(self):
        self.calls = 0

    def __call__(self, filename, data):
        self.calls += 1
        return process_pdf(filename, data)


PYTHON = ("python.pdf", make_pdf(PYTHON_PAGES))
OTHER = ("other.pdf", make_pdf(["A completely different document about cooking."]))


class SyncTests(unittest.TestCase):
    def test_same_file_is_processed_once(self):
        state, counter = {}, Counting()
        first = sync_learning_material(state, PYTHON, counter)
        second = sync_learning_material(state, PYTHON, counter)
        self.assertTrue(first.processed_now)
        self.assertFalse(second.processed_now)
        self.assertEqual(counter.calls, 1)
        self.assertIs(first.document, second.document)

    def test_replaced_file_leaves_no_stale_data(self):
        state = {}
        sync_learning_material(state, PYTHON)
        state[INDEX_KEY] = "old index"
        state[SEARCH_KEY] = "old results"
        status = sync_learning_material(state, OTHER)
        self.assertEqual(status.document.source_filename, "other.pdf")
        self.assertNotIn(INDEX_KEY, state)
        self.assertNotIn(SEARCH_KEY, state)

    def test_removed_file_clears_document(self):
        state = {}
        sync_learning_material(state, PYTHON)
        status = sync_learning_material(state, None)
        self.assertIsNone(status.document)
        self.assertNotIn(DOC_KEY, state)
        self.assertNotIn(DOC_STATUS_KEY, state)

    def test_failed_file_error_is_reused(self):
        state, counter = {}, Counting()
        bad = ("bad.pdf", b"not a pdf")
        first = sync_learning_material(state, bad, counter)
        second = sync_learning_material(state, bad, counter)
        self.assertIsNotNone(first.error)
        self.assertEqual(first.error, second.error)
        self.assertEqual(counter.calls, 1)
        self.assertNotIn(DOC_KEY, state)

    def test_unexpected_errors_show_generic_message(self):
        def exploding(filename, data):
            raise RuntimeError("secret internal detail")

        status = sync_learning_material({}, PYTHON, exploding)
        self.assertEqual(status.error, GENERIC_ERROR)
        self.assertNotIn("secret", status.error)

    def test_failed_upload_replaces_old_document(self):
        state = {}
        sync_learning_material(state, PYTHON)
        sync_learning_material(state, ("bad.pdf", b"nope"))
        self.assertNotIn(DOC_KEY, state)

    def test_clear_learning_material(self):
        state = {}
        sync_learning_material(state, PYTHON)
        clear_learning_material(state)
        self.assertEqual(state, {})


class CourseLinkTests(unittest.TestCase):
    def test_course_is_linked_to_pdf_loaded_at_generation(self):
        state = {}
        sync_learning_material(state, PYTHON)
        attach_material_to_course(state)
        self.assertIsNotNone(material_for_course(state))

    def test_later_pdf_is_not_the_courses_material(self):
        state = {}
        sync_learning_material(state, PYTHON)
        attach_material_to_course(state)
        sync_learning_material(state, OTHER)
        self.assertIsNone(material_for_course(state))

    def test_course_generated_without_pdf_has_no_material(self):
        state = {}
        attach_material_to_course(state)
        sync_learning_material(state, PYTHON)
        self.assertIsNone(state[COURSE_MATERIAL_KEY])
        self.assertIsNone(material_for_course(state))


if __name__ == "__main__":
    unittest.main()
