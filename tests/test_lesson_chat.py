import unittest

from classbuild.llm import ChatText, GenerationError
from classbuild.models import ChatMessage, Course, CourseSettings, validate_lesson
from classbuild.rag import EmbeddingService, build_index, process_pdf
from classbuild.services import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    ChatAnswerError,
    ChatInputError,
    LessonChatContext,
    LessonChatService,
    append_exchange,
    clear_thread,
    get_history,
    prune_chats,
    readable_citations,
    thread_key,
    valid_thread_keys,
)
from classbuild.services.lesson_chat import (
    MAX_HISTORY_MESSAGE_CHARS,
    MAX_HISTORY_TURNS,
    mentioned_pages,
    recent_history,
)
from tests.embedding_fakes import FakeEncoder
from tests.groq_fakes import PYTHON_PAGES, FakeGroq, course_data, lesson_data
from tests.pdf_fixtures import make_pdf

OUTLINE = Course.model_validate(course_data(2))
SETTINGS = CourseSettings("Python", "Beginner", 2)
LESSON = validate_lesson(lesson_data(1, "Lesson title 1"))
LESSON_2 = validate_lesson(lesson_data(2, "Lesson title 2"))
CTX = LessonChatContext.from_course(OUTLINE, SETTINGS, LESSON)


def pdf_reply(answer, ids, sufficient=True):
    return {"evidence_sufficient": sufficient, "answer": answer, "source_ids": ids}


class LessonModeTests(unittest.TestCase):
    def test_sends_lesson_and_bounded_history(self):
        history = []
        for n in range(10):
            history += [ChatMessage(role="user", content=f"q{n}"), ChatMessage(role="assistant", content=f"a{n}")]
        client = FakeGroq(chat_replies=["Simple answer."])
        reply = LessonChatService(client).answer(CTX, "Explain simply", history)
        messages = client.chat_calls[0]["messages"]
        self.assertIn("Intro text.", messages[0]["content"])  # the lesson is always sent
        self.assertEqual(messages[-1], {"role": "user", "content": "Explain simply"})
        self.assertEqual(len(messages), 1 + MAX_HISTORY_TURNS * 2 + 1)
        self.assertEqual(reply.answer.content, "Simple answer.")
        self.assertEqual(reply.answer.mode, "lesson")

    def test_truncated_answers_are_flagged(self):
        client = FakeGroq(chat_replies=[ChatText("cut off", "length")])
        self.assertTrue(LessonChatService(client).answer(CTX, "q").answer.truncated)

    def test_invalid_questions_never_reach_groq(self):
        client = FakeGroq(chat_replies=["x"])
        service = LessonChatService(client)
        for bad in ("", "   ", "x" * 1001):
            with self.assertRaises(ChatInputError):
                service.answer(CTX, bad)
        self.assertEqual(client.chat_calls, [])

    def test_groq_failure_leaves_history_untouched(self):
        chats, key = {}, thread_key(1, LESSON)
        client = FakeGroq(chat_replies=[GenerationError("down")])
        with self.assertRaises(GenerationError):
            LessonChatService(client).answer(CTX, "q", get_history(chats, key))
        self.assertEqual(get_history(chats, key), [])

    def test_recent_history_rules(self):
        history = [ChatMessage(role="assistant", content="orphan answer")]
        history += [ChatMessage(role="user", content="q"), ChatMessage(role="assistant", content="a" * 5000)]
        result = recent_history(history)
        self.assertEqual(result[0]["role"], "user")
        self.assertLessEqual(len(result[1]["content"]), MAX_HISTORY_MESSAGE_CHARS + 3)


class PdfModeTests(unittest.TestCase):
    def setUp(self):
        self.document = process_pdf("python.pdf", make_pdf(PYTHON_PAGES))
        self.service_embeddings = EmbeddingService(model=FakeEncoder())
        self.index = build_index(self.document, self.service_embeddings)

    def ask(self, client, question="What is the difference between lists and tuples?"):
        service = LessonChatService(client, self.service_embeddings)
        return service.answer(CTX, question, (), self.document, self.index)

    def test_valid_page_citations(self):
        client = FakeGroq(json_replies=[pdf_reply("Lists are mutable [p1-c1]; tuples are not.", ["p1-c1"])])
        reply = self.ask(client)
        self.assertEqual(reply.answer.status, "answered")
        self.assertEqual(reply.answer.cited_pages, [1])
        self.assertEqual(reply.answer.sources[0].page_number, 1)
        self.assertIn("p. 1", readable_citations(reply.answer.content, reply.answer.sources))

    def test_inline_marker_alone_counts_as_a_citation(self):
        reply = self.ask(FakeGroq(json_replies=[pdf_reply("Lists are mutable [p1-c1].", [])]))
        self.assertEqual(reply.answer.status, "answered")

    def test_unrelated_question_gets_fixed_reply_without_groq(self):
        client = FakeGroq(json_replies=[pdf_reply("made up", ["p1-c1"])])
        reply = self.ask(client, "What does the PDF say about Kubernetes pods?")
        self.assertEqual(reply.answer.status, "insufficient_evidence")
        self.assertEqual(reply.answer.content, INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertEqual(reply.answer.sources, [])
        self.assertEqual(client.json_calls, [])

    def test_model_says_evidence_insufficient(self):
        reply = self.ask(FakeGroq(json_replies=[pdf_reply("unsure", [], sufficient=False)]))
        self.assertEqual(reply.answer.status, "insufficient_evidence")

    def test_no_citations_gives_insufficient_reply(self):
        reply = self.ask(FakeGroq(json_replies=[pdf_reply("An uncited claim.", [])]))
        self.assertEqual(reply.answer.status, "insufficient_evidence")

    def test_invented_ids_are_rejected(self):
        with self.assertRaises(ChatAnswerError):
            self.ask(FakeGroq(json_replies=[pdf_reply("See [p9-c9].", ["p9-c9"])]))

    def test_uncited_page_mentions_are_rejected(self):
        with self.assertRaises(ChatAnswerError):
            self.ask(FakeGroq(json_replies=[pdf_reply("Lists are mutable, see page 2 [p1-c1].", ["p1-c1"])]))

    def test_pdf_mode_needs_an_index(self):
        with self.assertRaises(ChatInputError):
            LessonChatService(FakeGroq(), self.service_embeddings).answer(CTX, "q", (), self.document, None)

    def test_page_mention_detection(self):
        self.assertEqual(mentioned_pages("see page 3 and p. 4, also pp.5 but not [p3-c1]"), {3, 4, 5})


class ChatMessageTests(unittest.TestCase):
    def test_user_messages_have_no_sources(self):
        self.assertEqual(ChatMessage(role="user", content="hi").sources, [])

    def test_assistant_lesson_answer_is_valid(self):
        self.assertEqual(ChatMessage(role="assistant", content="hello").mode, "lesson")

    def test_empty_content_is_rejected(self):
        with self.assertRaises(ValueError):
            ChatMessage(role="user", content="   ")

    def test_answered_pdf_reply_needs_sources(self):
        with self.assertRaises(ValueError):
            ChatMessage(role="assistant", content="x", mode="pdf", document_fingerprint="f")


class ThreadStateTests(unittest.TestCase):
    def exchange(self, text="q"):
        return ChatMessage(role="user", content=text), ChatMessage(role="assistant", content="a " + text)

    def test_each_lesson_has_its_own_history(self):
        chats = {}
        key1, key2 = thread_key(1, LESSON), thread_key(1, LESSON_2)
        append_exchange(chats, key1, *self.exchange("one"))
        append_exchange(chats, key2, *self.exchange("two"))
        self.assertEqual([m.content for m in get_history(chats, key1)], ["one", "a one"])
        self.assertEqual([m.content for m in get_history(chats, key2)], ["two", "a two"])

    def test_keys_separate_courses_lessons_modes_and_versions(self):
        keys = {
            thread_key(1, LESSON),
            thread_key(2, LESSON),
            thread_key(1, LESSON_2),
            thread_key(1, LESSON, "pdf-a"),
            thread_key(1, LESSON, "pdf-b"),
            thread_key(1, validate_lesson({**lesson_data(1, "Lesson title 1"), "summary": "Changed"})),
        }
        self.assertEqual(len(keys), 6)
        self.assertTrue(thread_key(1, LESSON, "pdf-a").is_pdf)
        self.assertFalse(thread_key(1, LESSON).is_pdf)

    def test_clear_thread_only_clears_one(self):
        chats = {}
        key1, key2 = thread_key(1, LESSON), thread_key(1, LESSON_2)
        append_exchange(chats, key1, *self.exchange())
        append_exchange(chats, key2, *self.exchange())
        clear_thread(chats, key1)
        self.assertEqual(get_history(chats, key1), [])
        self.assertEqual(len(get_history(chats, key2)), 2)

    def test_pdf_replacement_prunes_only_pdf_threads(self):
        chats = {}
        lesson_key, old_pdf_key = thread_key(1, LESSON), thread_key(1, LESSON, "old-pdf")
        append_exchange(chats, lesson_key, *self.exchange())
        append_exchange(chats, old_pdf_key, *self.exchange())
        valid = valid_thread_keys(1, {1: LESSON}, "new-pdf")
        self.assertEqual(prune_chats(chats, valid), 1)
        self.assertIn(lesson_key, chats)
        self.assertNotIn(old_pdf_key, chats)

    def test_new_course_prunes_old_course_threads(self):
        chats = {}
        append_exchange(chats, thread_key(1, LESSON), *self.exchange())
        self.assertEqual(prune_chats(chats, valid_thread_keys(2, {1: LESSON}, None)), 1)

    def test_only_complete_exchanges_are_appended(self):
        with self.assertRaises(ValueError):
            append_exchange({}, thread_key(1, LESSON), *reversed(self.exchange()))


if __name__ == "__main__":
    unittest.main()
