"""Find and remove grounded lessons whose PDF is no longer the loaded one."""

from classbuild.models import LessonContent


def stale_grounded_lessons(lessons: dict[int, LessonContent], current_fingerprint: str | None) -> list[int]:
    """Numbers of grounded lessons whose PDF fingerprint differs from the current one."""
    return sorted(
        number
        for number, lesson in lessons.items()
        if lesson.grounding is not None and lesson.grounding.document_fingerprint != current_fingerprint
    )


def remove_stale_grounded_lessons(
    lessons: dict[int, LessonContent], current_fingerprint: str | None, *dependent: dict
) -> list[int]:
    """Remove stale grounded lessons and their entries in the dependent dicts
    (quizzes, results, attempts). Ordinary lessons are never removed."""
    stale = stale_grounded_lessons(lessons, current_fingerprint)
    for number in stale:
        lessons.pop(number, None)
        for mapping in dependent:
            mapping.pop(number, None)
    return stale
