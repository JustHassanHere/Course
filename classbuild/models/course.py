from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Lesson(BaseModel):
    title: NonEmptyStr
    objectives: list[NonEmptyStr] = Field(min_length=1)
    summary: NonEmptyStr


class Course(BaseModel):
    title: NonEmptyStr
    description: NonEmptyStr
    objectives: list[NonEmptyStr] = Field(min_length=1)
    lessons: list[Lesson] = Field(min_length=1)


@dataclass(frozen=True)
class CourseSettings:
    """The settings a course was generated with."""

    topic: str
    difficulty: str
    lesson_count: int


def validate_course(data: dict, lesson_count: int) -> Course:
    course = Course.model_validate(data)
    if len(course.lessons) != lesson_count:
        raise ValueError(f"Expected exactly {lesson_count} lessons but got {len(course.lessons)}.")
    return course
