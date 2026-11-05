"""Step 3: generate and validate a course outline."""

from classbuild.models import Course, CourseSettings, validate_course
from classbuild.services.structured_output import BASE_SYSTEM_PROMPT, request_structured_json, to_strict_schema


def course_json_schema() -> dict:
    return to_strict_schema(Course.model_json_schema())


def build_course_prompt(settings: CourseSettings) -> str:
    return f"""Create a course outline.
Topic: {settings.topic}
Difficulty: {settings.difficulty}
Number of lessons: exactly {settings.lesson_count}

Each lesson needs a title, learning objectives and a short summary.
The "lessons" array must contain exactly {settings.lesson_count} items."""


def generate_course(client, settings: CourseSettings) -> Course:
    return request_structured_json(
        client,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=build_course_prompt(settings),
        schema=course_json_schema(),
        schema_name="course_outline",
        validate=lambda data: validate_course(data, settings.lesson_count),
        label="course outline",
    )
