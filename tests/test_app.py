"""UI flows using Streamlit's AppTest with a fake Groq client and a fake encoder."""

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import classbuild.services  # noqa: F401  (import before patching so annotations see the real class)
from classbuild.rag import EmbeddingService as RealEmbeddingService
from classbuild.rag import process_pdf
from tests.embedding_fakes import FakeEncoder
from tests.groq_fakes import PYTHON_PAGES, FakeGroq, grounded_draft, lesson_data, quiz_data
from tests.pdf_fixtures import make_pdf

APP = str(Path(__file__).resolve().parents[1] / "app_stage2.py")

COURSE = {
    "title": "Container orchestration",
    "description": "A first course.",
    "objectives": ["Learn the basics"],
    "lessons": [
        {
            "title": "Lists and tuples",
            "objectives": ["Explain mutable lists and immutable tuples"],
            "summary": "How Python lists and tuples differ",
        },
        {
            "title": "Kubernetes pods",
            "objectives": ["Describe Kubernetes pods and cluster"],
            "summary": "Kubernetes pod and container basics",
        },
    ],
}


class Upload:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


PYTHON_UPLOAD = Upload("python.pdf", make_pdf(PYTHON_PAGES))
OTHER_UPLOAD = Upload("other.pdf", make_pdf(["A document about python strings and string loops."]))


def button(at, label):
    return next(b for b in at.button if b.label == label)


def has_button(at, label):
    return any(b.label == label for b in at.button)


class AppCase(unittest.TestCase):
    def setUp(self):
        self.client = FakeGroq()
        self.encoder = FakeEncoder()
        self.upload = None
        self.process_calls = 0
        patches = [
            patch("classbuild.llm.get_client", lambda: self.client),
            patch("classbuild.rag.EmbeddingService", lambda *a, **k: RealEmbeddingService(model=self.encoder)),
            patch("streamlit.file_uploader", lambda *a, **k: self.upload),
            patch("classbuild.rag.document_state.process_pdf", self.counting_process),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.at = AppTest.from_file(APP, default_timeout=30)

    def counting_process(self, filename, data):
        self.process_calls += 1
        return process_pdf(filename, data)

    # ---- helpers ----------------------------------------------------------------
    def run_app(self):
        self.at.run()
        self.assertFalse(self.at.exception, [e.value for e in self.at.exception])
        return self.at

    def generate_course(self, topic="Introduction to Python"):
        self.run_app()  # widgets exist only after a first run
        self.at.text_input(key="topic_input").set_value(topic)
        button(self.at, "Generate Course Outline").click()
        return self.run_app()

    def select_lesson(self, number):
        self.at.selectbox(key="lesson_picker").set_value(number - 1)
        return self.run_app()

    def click(self, label):
        button(self.at, label).click()
        return self.run_app()

    def texts(self, element_list):
        return " ".join(str(e.value) for e in element_list)


class BasicFlowTests(AppCase):
    def test_ui_works_without_course_or_pdf(self):
        at = self.run_app()
        self.assertIn("No course yet", self.texts(at.info))
        self.assertNotIn("Use Uploaded PDF as Reference", [c.label for c in at.checkbox])
        self.assertIn("No PDF loaded", self.texts(at.caption))

    def test_empty_topic_warns_and_does_not_call_groq(self):
        self.run_app()
        self.at.text_input(key="topic_input").set_value("   ")
        at = self.click("Generate Course Outline")
        self.assertIn("enter a course topic", self.texts(at.warning))
        self.assertEqual(self.client.json_calls, [])

    def test_course_lesson_quiz_and_stable_reruns(self):
        self.client.json_replies = [COURSE, lesson_data(1, "Lists and tuples"), quiz_data()]
        self.generate_course()
        self.assertEqual(len(self.client.json_calls), 1)
        self.click("Generate Lesson Content")
        self.assertEqual(len(self.client.json_calls), 2)
        self.assertIn("Intro text.", self.texts(self.at.markdown))
        self.at.text_input(key="topic_input").set_value("another topic")  # unrelated widget
        self.run_app()
        self.assertEqual(len(self.client.json_calls), 2)  # no new request
        self.assertIn("Intro text.", self.texts(self.at.markdown))
        self.click("Generate Quiz")
        self.assertEqual(len(self.at.radio), 5)

    def test_new_course_clears_old_lessons(self):
        self.client.json_replies = [COURSE, lesson_data(1, "Lists and tuples"), COURSE]
        self.generate_course()
        self.click("Generate Lesson Content")
        self.generate_course("Second topic")
        self.assertNotIn("Intro text.", self.texts(self.at.markdown))
        self.assertTrue(has_button(self.at, "Generate Lesson Content"))


class PdfUploadTests(AppCase):
    def test_valid_pdf_is_processed_once(self):
        self.upload = PYTHON_UPLOAD
        at = self.run_app()
        self.assertIn("Extracted text from 2 of 2 pages", self.texts(at.success))
        self.run_app()
        self.run_app()
        self.assertEqual(self.process_calls, 1)

    def test_invalid_file_shows_error_not_crash(self):
        self.upload = Upload("fake.pdf", b"this is plain text, not a pdf")
        at = self.run_app()
        self.assertTrue(at.error)
        self.assertNotIn("Extracted text", self.texts(at.success))

    def test_image_only_pdf_reports_no_text(self):
        self.upload = Upload("scan.pdf", make_pdf([None, None]))
        self.assertIn("scanned", self.texts(self.run_app().error))

    def test_removing_pdf_clears_state(self):
        self.upload = PYTHON_UPLOAD
        self.run_app()
        self.upload = None
        at = self.run_app()
        self.assertIn("No PDF loaded", self.texts(at.caption))
        self.assertNotIn("pdf_document", at.session_state)

    def test_search_returns_ranked_results_with_pages(self):
        self.upload = PYTHON_UPLOAD
        self.run_app()
        self.at.text_input(key=next(k for k in self._keys() if k.startswith("pdf_search_query_"))).set_value(
            "difference between lists and tuples mutable immutable"
        )
        at = self.click("Search PDF")
        self.assertIn("page 1", self.texts(at.markdown))
        before = self.encoder.texts_encoded
        self.run_app()
        self.assertEqual(self.encoder.texts_encoded, before)  # nothing recomputed on rerun

    def _keys(self):
        return [t.key for t in self.at.text_input]


class GroundedLessonTests(AppCase):
    def setup_course(self):
        self.upload = PYTHON_UPLOAD
        self.run_app()
        self.generate_course()
        self.at.checkbox(key=self._pdf_key()).set_value(True)
        self.run_app()

    def _pdf_key(self):
        return next(c.key for c in self.at.checkbox if c.label == "Use Uploaded PDF as Reference")

    def test_grounded_lesson_shows_sources_and_pages(self):
        self.client.json_replies = [COURSE, grounded_draft(1, "Lists and tuples", ["p1-c1"])]
        self.setup_course()
        self.click("Generate Lesson Content")
        text = self.texts(self.at.markdown) + self.texts(self.at.caption)
        self.assertIn("Sources", text)
        self.assertIn("p1-c1", text)
        self.assertIn("page 1", text)

    def test_unrelated_lesson_reports_insufficient_evidence(self):
        self.client.json_replies = [COURSE, lesson_data(2, "Kubernetes pods")]
        self.setup_course()
        self.select_lesson(2)
        self.click("Generate Lesson Content")
        self.assertIn("not generated from the PDF", self.texts(self.at.warning))
        self.assertEqual(len(self.client.json_calls), 1)  # only the course request
        self.click("Generate Without PDF")
        self.assertIn("Intro text.", self.texts(self.at.markdown))
        self.assertFalse(any("Sources" == m.value for m in self.at.markdown))

    def test_ordinary_generation_with_option_off(self):
        self.client.json_replies = [COURSE, lesson_data(1, "Lists and tuples")]
        self.upload = PYTHON_UPLOAD
        self.run_app()
        self.generate_course()
        self.click("Generate Lesson Content")
        self.assertIn("Intro text.", self.texts(self.at.markdown))
        self.assertNotIn("Based on", self.texts(self.at.caption))

    def test_replacing_pdf_removes_grounded_lessons(self):
        self.client.json_replies = [COURSE, grounded_draft(1, "Lists and tuples", ["p1-c1"])]
        self.setup_course()
        self.click("Generate Lesson Content")
        self.upload = OTHER_UPLOAD
        at = self.run_app()
        self.assertIn("no longer loaded", self.texts(at.warning))
        self.assertTrue(has_button(at, "Generate Lesson Content"))


class ChatTests(AppCase):
    def prepare(self, extra=()):
        self.client.json_replies = [
            COURSE,
            lesson_data(1, "Lists and tuples"),
            lesson_data(2, "Kubernetes pods"),
            *extra,
        ]
        self.generate_course()
        self.click("Generate Lesson Content")

    def ask(self, text):
        self.at.chat_input[0].set_value(text).run()
        self.assertFalse(self.at.exception, [e.value for e in self.at.exception])

    def messages(self):
        return [m.markdown[0].value for m in self.at.chat_message]

    def test_conversation_is_kept_across_reruns(self):
        self.client.chat_replies = ["Simpler words.", "Another example."]
        self.prepare()
        self.ask("Explain this lesson in simpler words")
        self.ask("Can you give me another example?")
        self.assertEqual(
            self.messages(),
            ["Explain this lesson in simpler words", "Simpler words.", "Can you give me another example?", "Another example."],
        )
        sent = self.client.chat_calls[1]["messages"]
        self.assertEqual(sent[1]["content"], "Explain this lesson in simpler words")  # history included
        self.run_app()
        self.assertEqual(len(self.messages()), 4)

    def test_each_lesson_keeps_its_own_chat_and_clear_chat_is_local(self):
        self.client.chat_replies = ["Answer for one.", "Answer for two."]
        self.prepare()
        self.ask("Question one")
        self.select_lesson(2)
        self.click("Generate Lesson Content")
        self.ask("Question two")
        self.assertEqual(self.messages(), ["Question two", "Answer for two."])
        self.select_lesson(1)
        self.assertEqual(self.messages(), ["Question one", "Answer for one."])
        self.click("Clear Chat")
        self.assertEqual(self.messages(), [])
        self.select_lesson(2)
        self.assertEqual(self.messages(), ["Question two", "Answer for two."])

    def test_blank_question_warns_without_calling_groq(self):
        self.prepare()
        self.ask("   ")
        self.assertIn("type a question", self.texts(self.at.warning))
        self.assertEqual(self.client.chat_calls, [])
        self.assertEqual(self.messages(), [])

    def test_groq_failure_does_not_corrupt_history(self):
        from classbuild.llm import GenerationError

        self.client.chat_replies = ["Fine answer.", GenerationError("Groq is down")]
        self.prepare()
        self.ask("First question")
        self.ask("Second question")
        self.assertIn("Groq is down", self.texts(self.at.error))
        self.assertEqual(self.messages(), ["First question", "Fine answer."])

    def test_pdf_mode_cites_pages_and_refuses_without_evidence(self):
        self.upload = PYTHON_UPLOAD
        self.run_app()
        self.prepare(
            extra=[
                {"evidence_sufficient": True, "answer": "Lists are mutable [p1-c1].", "source_ids": ["p1-c1"]},
            ]
        )
        key = next(c.key for c in self.at.checkbox if c.label == "Answer from the uploaded PDF")
        self.at.checkbox(key=key).set_value(True)
        self.run_app()
        json_calls = len(self.client.json_calls)
        self.client.json_replies = [
            {"evidence_sufficient": True, "answer": "Lists are mutable [p1-c1].", "source_ids": ["p1-c1"]}
        ]
        self.ask("What is the difference between lists and tuples?")
        self.assertIn("[p. 1]", self.messages()[-1])
        self.assertIn("p. 1", self.texts(self.at.caption))
        self.ask("What does the PDF say about Kubernetes pods?")
        self.assertIn("does not contain enough evidence", self.messages()[-1])
        self.assertEqual(len(self.client.json_calls), json_calls + 1)  # the refusal made no Groq call


if __name__ == "__main__":
    unittest.main()
