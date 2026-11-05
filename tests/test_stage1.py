"""Stage 1 behaviour in the classbuild package: models, generators, grading, client."""

import json
import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from classbuild.llm import ChatText, ConfigError, GenerationError, GroqClient
from classbuild.models import Course, CourseSettings, LessonContent, Quiz, validate_lesson
from classbuild.services import generate_course, generate_lesson, generate_quiz, grade_quiz
from classbuild.services.lesson_generator import lesson_json_schema
from classbuild.services.structured_output import to_strict_schema
from tests.groq_fakes import FakeGroq, course_data, lesson_data, quiz_data

SETTINGS = CourseSettings("Introduction to Python", "Beginner", 3)


class CourseTests(unittest.TestCase):
    def test_exact_lesson_counts(self):
        for count in (2, 3, 5):
            settings = CourseSettings("Python", "Beginner", count)
            client = FakeGroq(json_replies=[course_data(count)])
            self.assertEqual(len(generate_course(client, settings).lessons), count)

    def test_missing_title_is_rejected_then_controlled_error(self):
        bad = course_data(3)
        del bad["title"]
        client = FakeGroq(json_replies=[bad])
        with self.assertRaises(GenerationError):
            generate_course(client, SETTINGS)
        self.assertEqual(len(client.json_calls), 2)  # one retry only

    def test_wrong_lesson_count_is_never_accepted(self):
        with self.assertRaises(GenerationError):
            generate_course(FakeGroq(json_replies=[course_data(4)]), SETTINGS)

    def test_retry_recovers_from_invalid_json(self):
        client = FakeGroq(json_replies=["not json at all", course_data(3)])
        self.assertEqual(len(generate_course(client, SETTINGS).lessons), 3)
        self.assertIn("rejected", client.json_calls[1]["user"])

    def test_code_fences_are_accepted(self):
        client = FakeGroq(json_replies=["```json\n" + json.dumps(course_data(3)) + "\n```"])
        self.assertEqual(len(generate_course(client, SETTINGS).lessons), 3)

    def test_schema_is_strict_and_keeps_description_property(self):
        schema = to_strict_schema(Course.model_json_schema())
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("description", schema["properties"])
        self.assertEqual(schema["required"], list(schema["properties"]))
        self.assertNotIn("$ref", json.dumps(schema))


class LessonTests(unittest.TestCase):
    def setUp(self):
        self.outline = Course.model_validate(course_data(3))

    def test_valid_lesson(self):
        client = FakeGroq(json_replies=[lesson_data(2, "Lesson title 2")])
        lesson = generate_lesson(client, self.outline, SETTINGS, 2)
        self.assertEqual(len(lesson.exercises), 2)
        self.assertFalse(lesson.is_grounded)

    def test_schema_has_no_citation_fields_and_pins_identity(self):
        schema = lesson_json_schema(2, self.outline.lessons[1])
        text = json.dumps(schema)
        self.assertNotIn("source_ids", text)
        self.assertNotIn("grounding", text)
        self.assertEqual(schema["properties"]["lesson_number"]["enum"], [2])
        self.assertEqual(schema["properties"]["title"]["enum"], ["Lesson title 2"])

    def test_wrong_number_or_title_is_rejected(self):
        for reply in (lesson_data(1, "Lesson title 2"), lesson_data(2, "Another title")):
            with self.assertRaises(GenerationError):
                generate_lesson(FakeGroq(json_replies=[reply]), self.outline, SETTINGS, 2)

    def test_section_and_exercise_limits(self):
        data = lesson_data()
        data["sections"] = data["sections"][:1]
        with self.assertRaises(ValidationError):
            LessonContent.model_validate(data)
        data = lesson_data()
        data["exercises"] = data["exercises"][:1]
        with self.assertRaises(ValidationError):
            LessonContent.model_validate(data)
        data = lesson_data()
        del data["exercises"][0]["hint"]
        with self.assertRaises(ValidationError):
            LessonContent.model_validate(data)

    def test_ordinary_lesson_cannot_contain_citations(self):
        data = lesson_data()
        data["sections"][0]["source_ids"] = ["p1-c1"]
        with self.assertRaises(ValidationError):
            validate_lesson(data)


class QuizTests(unittest.TestCase):
    def setUp(self):
        self.outline = Course.model_validate(course_data(3))

    def test_quiz_schema_rules(self):
        self.assertEqual(len(Quiz.model_validate(quiz_data()).questions), 5)
        mutations = [
            lambda d: d["questions"].pop(),
            lambda d: d["questions"][0]["options"].pop(),
            lambda d: d["questions"][0].update(correct_index=4),
            lambda d: d["questions"][0].update(options=["a", "a", "c", "d"]),
        ]
        for mutate in mutations:
            data = quiz_data()
            mutate(data)
            with self.assertRaises(ValidationError):
                Quiz.model_validate(data)

    def test_generate_quiz(self):
        quiz = generate_quiz(FakeGroq(json_replies=[quiz_data()]), self.outline, SETTINGS, 1)
        self.assertEqual(len(quiz.questions), 5)

    def test_local_grading(self):
        quiz = Quiz.model_validate(quiz_data())
        perfect = [q.correct_index for q in quiz.questions]
        self.assertEqual(grade_quiz(quiz, perfect)["score"], 5)
        self.assertEqual(grade_quiz(quiz, [(a + 1) % 4 for a in perfect])["score"], 0)
        mixed = list(perfect)
        mixed[0] = (mixed[0] + 1) % 4
        self.assertEqual(grade_quiz(quiz, mixed)["score"], 4)
        with self.assertRaises(ValueError):
            grade_quiz(quiz, [0, 1, None, 0, 1])


class FakeSdk:
    """Imitates groq.Groq().chat.completions.create."""

    def __init__(self, content="reply", finish_reason="stop", fail_with=None):
        self.requests = []
        self._content, self._finish, self._fail = content, finish_reason, fail_with
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self._fail and kwargs.get("response_format") is not None:
            raise self._fail
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self._finish)])


class GroqClientTests(unittest.TestCase):
    def test_missing_key_gives_clear_error(self):
        with self.assertRaises(ConfigError):
            GroqClient(api_key="")

    def test_chat_requires_a_final_user_message(self):
        client = GroqClient(model="m", sdk_client=FakeSdk())
        for messages in ([], [{"role": "assistant", "content": "hi"}], [{"role": "user", "content": "  "}]):
            with self.assertRaises(ValueError):
                client.chat(messages)

    def test_chat_returns_text_and_finish_reason(self):
        sdk = FakeSdk(content="hello", finish_reason="length")
        result = GroqClient(model="m", sdk_client=sdk).chat(
            [{"role": "user", "content": "hi"}], max_completion_tokens=50
        )
        self.assertEqual(result, ChatText(text="hello", finish_reason="length"))
        self.assertEqual(sdk.requests[0]["max_completion_tokens"], 50)

    def test_reasoning_effort_only_for_reasoning_models(self):
        sdk = FakeSdk()
        GroqClient(model="openai/gpt-oss-20b", sdk_client=sdk).chat(
            [{"role": "user", "content": "hi"}], reasoning_effort="low"
        )
        self.assertEqual(sdk.requests[0]["extra_body"], {"reasoning_effort": "low"})
        sdk = FakeSdk()
        GroqClient(model="some-other-model", sdk_client=sdk).chat(
            [{"role": "user", "content": "hi"}], reasoning_effort="low"
        )
        self.assertNotIn("extra_body", sdk.requests[0])

    def test_generate_json_places_history_between_system_and_user(self):
        sdk = FakeSdk(content="{}")
        history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
        GroqClient(model="m", sdk_client=sdk).generate_json("SYS", "NEW", schema={"type": "object"}, history=history)
        roles = [m["role"] for m in sdk.requests[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(sdk.requests[0]["response_format"]["type"], "json_schema")

    def test_schema_rejection_falls_back_to_prompt_schema(self):
        from groq import BadRequestError

        response = SimpleNamespace(status_code=400, request=SimpleNamespace(), headers={})
        error = BadRequestError("bad schema", response=response, body=None)
        sdk = FakeSdk(content="{}", fail_with=error)
        GroqClient(model="m", sdk_client=sdk).generate_json("SYS", "NEW", schema={"type": "object"})
        self.assertEqual(len(sdk.requests), 2)
        self.assertNotIn("response_format", sdk.requests[1])
        self.assertIn("JSON schema", sdk.requests[1]["messages"][0]["content"])

    def test_empty_reply_is_an_error(self):
        with self.assertRaises(GenerationError):
            GroqClient(model="m", sdk_client=FakeSdk(content="  ")).chat([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
