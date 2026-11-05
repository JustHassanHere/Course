"""Step 5: quiz generation and local grading."""

from classbuild.models import Course, CourseSettings, LessonContent, Quiz
from classbuild.services.lesson_generator import find_lesson
from classbuild.services.structured_output import BASE_SYSTEM_PROMPT, request_structured_json, to_strict_schema


def quiz_json_schema() -> dict:
    return to_strict_schema(Quiz.model_json_schema())


def build_quiz_prompt(
    outline: Course, settings: CourseSettings, lesson_number: int, content: LessonContent | None
) -> str:
    lesson = find_lesson(outline, lesson_number)
    objectives = "\n".join(f"- {objective}" for objective in lesson.objectives)
    covered = ""
    if content is not None:
        covered = "\nSections covered in the lesson: " + ", ".join(s.heading for s in content.sections)
    return f"""Write a multiple-choice quiz for one lesson.
Course topic: {settings.topic}
Course title: {outline.title}
Difficulty: {settings.difficulty}
Lesson title: {lesson.title}
Lesson summary: {lesson.summary}
Lesson objectives:
{objectives}{covered}

Rules: exactly 5 questions; each has exactly 4 distinct options; "correct_index" is the
zero-based index (0-3) of the single correct option; vary the position of the correct
option across questions; add a short explanation to every question."""


def generate_quiz(
    client, outline: Course, settings: CourseSettings, lesson_number: int, content: LessonContent | None = None
) -> Quiz:
    return request_structured_json(
        client,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=build_quiz_prompt(outline, settings, lesson_number, content),
        schema=quiz_json_schema(),
        schema_name="quiz",
        validate=Quiz.model_validate,
        label="quiz",
    )


def grade_quiz(quiz: Quiz, answers: list[int | None]) -> dict:
    """Return the score (0-5) and per-question results. Graded locally, no model request."""
    if len(answers) != len(quiz.questions):
        raise ValueError("One answer is required for every question.")
    if any(answer is None for answer in answers):
        raise ValueError("All questions must be answered before grading.")
    correct = [answer == q.correct_index for answer, q in zip(answers, quiz.questions)]
    return {"score": sum(correct), "answers": list(answers), "correct": correct}
