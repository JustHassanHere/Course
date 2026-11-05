"""Chat histories, one conversation per lesson (and per mode)."""

import hashlib
from typing import NamedTuple

from classbuild.models import ChatMessage, LessonContent


class ChatThreadKey(NamedTuple):
    course_id: int
    lesson_number: int
    lesson_fingerprint: str
    document_fingerprint: str | None  # None in lesson mode

    @property
    def is_pdf(self) -> bool:
        return self.document_fingerprint is not None


def lesson_fingerprint(content: LessonContent) -> str:
    return hashlib.sha256(content.model_dump_json().encode("utf-8")).hexdigest()[:16]


def thread_key(course_id: int, content: LessonContent, document_fingerprint: str | None = None) -> ChatThreadKey:
    return ChatThreadKey(course_id, content.lesson_number, lesson_fingerprint(content), document_fingerprint)


def get_history(chats: dict, key: ChatThreadKey) -> list[ChatMessage]:
    return list(chats.get(key, []))


def append_exchange(chats: dict, key: ChatThreadKey, question: ChatMessage, answer: ChatMessage) -> None:
    """Add a question and its answer together, nothing else."""
    if question.role != "user" or answer.role != "assistant":
        raise ValueError("An exchange is a user question followed by an assistant answer.")
    chats.setdefault(key, []).extend([question, answer])


def clear_thread(chats: dict, key: ChatThreadKey) -> None:
    chats.pop(key, None)


def valid_thread_keys(
    course_id: int, lessons: dict[int, LessonContent], document_fingerprint: str | None
) -> set[ChatThreadKey]:
    keys = set()
    for content in lessons.values():
        keys.add(thread_key(course_id, content, None))
        if document_fingerprint is not None:
            keys.add(thread_key(course_id, content, document_fingerprint))
    return keys


def prune_chats(chats: dict, valid_keys: set[ChatThreadKey]) -> int:
    """Delete every thread that is not valid any more; returns how many were removed."""
    stale = [key for key in chats if key not in valid_keys]
    for key in stale:
        del chats[key]
    return len(stale)
