"""A fake Groq client plus small data builders used by several tests."""

import json

from classbuild.llm import ChatText


class FakeGroq:
    """Replies are consumed in order; the last one repeats. A reply may be a dict, a string,
    a ChatText (chat only) or an Exception instance (raised)."""

    def __init__(self, json_replies=None, chat_replies=None):
        self.json_replies = list(json_replies or [])
        self.chat_replies = list(chat_replies or [])
        self.json_calls = []
        self.chat_calls = []
        self.model = "fake-model"

    @staticmethod
    def _next(replies):
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def generate_json(self, system_prompt, user_prompt, schema=None, schema_name="response", history=None, **kwargs):
        self.json_calls.append(
            {"system": system_prompt, "user": user_prompt, "schema": schema, "name": schema_name, "history": history}
        )
        reply = self._next(self.json_replies)
        return reply if isinstance(reply, str) else json.dumps(reply)

    def chat(self, messages, max_completion_tokens=None, reasoning_effort=None, **kwargs):
        self.chat_calls.append({"messages": messages, "max": max_completion_tokens})
        reply = self._next(self.chat_replies)
        return reply if isinstance(reply, ChatText) else ChatText(text=reply, finish_reason="stop")


def course_data(count=3):
    return {
        "title": "Intro to Python",
        "description": "A first course.",
        "objectives": ["Write simple programs"],
        "lessons": [
            {
                "title": f"Lesson title {i}",
                "objectives": [f"Objective {i}"],
                "summary": f"Summary {i}",
            }
            for i in range(1, count + 1)
        ],
    }


def lesson_data(number=1, title="Lesson title 1"):
    return {
        "lesson_number": number,
        "title": title,
        "introduction": "Intro text.",
        "sections": [
            {"heading": "First", "explanation": "Explanation one.", "example": "Example one."},
            {"heading": "Second", "explanation": "Explanation two.", "example": "Example two."},
        ],
        "summary": "Summary text.",
        "exercises": [
            {"task": "Task one", "hint": "Hint one"},
            {"task": "Task two", "hint": "Hint two"},
        ],
    }


def quiz_data():
    return {
        "questions": [
            {
                "question": f"Question {i}?",
                "options": ["A", "B", "C", "D"],
                "correct_index": i % 4,
                "explanation": f"Because {i}.",
            }
            for i in range(5)
        ]
    }


def grounded_draft(number, title, source_ids, relevant=True):
    """A model reply for a grounded lesson; every section cites `source_ids`."""
    return {
        "evidence_relevant": relevant,
        "lesson_number": number,
        "title": title,
        "introduction": "Intro text.",
        "sections": [
            {"source_ids": list(source_ids), "heading": "First", "explanation": "E1.", "example": "X1."},
            {"source_ids": list(source_ids)[:1], "heading": "Second", "explanation": "E2.", "example": "X2."},
        ],
        "summary": "Summary text.",
        "exercises": [{"task": "T1", "hint": "H1"}, {"task": "T2", "hint": "H2"}],
    }


PYTHON_PAGES = [
    (
        "Python is dynamically typed. A variable can refer to a value of any type, and the type "
        "is checked while the program runs.\n\n"
        "A list is mutable, so you can change a list after creating it. A tuple is immutable, so "
        "a tuple cannot change. Lists and tuples both store ordered values."
    ),
    (
        "A function groups statements under a name. Python functions are defined with the def "
        "keyword and use indentation.\n\n"
        "A loop repeats statements. Python has a for loop and a while loop; strings can be "
        "looped over one character at a time."
    ),
]
