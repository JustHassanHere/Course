"""Step 4: ordinary (not PDF-grounded) lesson generation, plus lesson helpers."""

import copy

from classbuild.models import Course, CourseSettings, Lesson, LessonContent, validate_lesson
from classbuild.services.structured_output import BASE_SYSTEM_PROMPT, request_structured_json, to_strict_schema


def find_lesson(outline: Course, lesson_number: int) -> Lesson:
    """Lessons are numbered from 1."""
    if not 1 <= lesson_number <= len(outline.lessons):
        raise ValueError(f"Lesson {lesson_number} does not exist in this course.")
    return outline.lessons[lesson_number - 1]


def _without_source_fields(schema: dict) -> dict:
    """Remove PDF citation fields: ordinary lessons must never contain citations."""
    schema = copy.deepcopy(schema)
    schema.get("properties", {}).pop("grounding", None)
    if "required" in schema:
        schema["required"] = [name for name in schema["required"] if name != "grounding"]
    definitions = schema.get("$defs", {})
    definitions.pop("LessonGrounding", None)
    definitions.pop("SourceReference", None)
    section = definitions.get("LessonSection", {})
    section.get("properties", {}).pop("source_ids", None)
    if "required" in section:
        section["required"] = [name for name in section["required"] if name != "source_ids"]
    return schema


def pin_lesson_identity(strict_schema: dict, lesson_number: int, title: str) -> dict:
    """Pin lesson number and title so the model cannot change them."""
    strict_schema["properties"]["lesson_number"] = {"type": "integer", "enum": [lesson_number]}
    strict_schema["properties"]["title"] = {"type": "string", "enum": [title]}
    return strict_schema


def lesson_json_schema(lesson_number: int, lesson: Lesson) -> dict:
    base = _without_source_fields(LessonContent.model_json_schema())
    return pin_lesson_identity(to_strict_schema(base), lesson_number, lesson.title)


def build_lesson_prompt(outline: Course, settings: CourseSettings, lesson_number: int) -> str:
    lesson = find_lesson(outline, lesson_number)
    objectives = "\n".join(f"- {objective}" for objective in lesson.objectives)
    return f"""Write the full content of one lesson.
Course topic: {settings.topic}
Course title: {outline.title}
Difficulty: {settings.difficulty}
Lesson number: {lesson_number}
Lesson title: {lesson.title}
Lesson summary: {lesson.summary}
Lesson objectives:
{objectives}

Rules: use lesson_number {lesson_number} and the exact lesson title; write an introduction,
2 to 4 sections (heading, explanation, example), a summary, and exactly 2 exercises with a
task and a hint each."""


def generate_lesson(client, outline: Course, settings: CourseSettings, lesson_number: int) -> LessonContent:
    lesson = find_lesson(outline, lesson_number)
    return request_structured_json(
        client,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=build_lesson_prompt(outline, settings, lesson_number),
        schema=lesson_json_schema(lesson_number, lesson),
        schema_name="lesson",
        validate=lambda data: validate_lesson(
            data, expected_lesson_number=lesson_number, expected_title=lesson.title
        ),
        label="lesson",
    )
