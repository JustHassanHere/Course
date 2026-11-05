from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from classbuild.models.course import NonEmptyStr


class SourceReference(BaseModel):
    """A cited PDF passage. Filled from retrieval results, never from the AI."""

    chunk_id: NonEmptyStr
    source_filename: NonEmptyStr
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)
    similarity: float
    text: NonEmptyStr


class LessonSection(BaseModel):
    source_ids: list[str] = Field(default_factory=list)
    heading: NonEmptyStr
    explanation: NonEmptyStr
    example: NonEmptyStr

    @field_validator("source_ids")
    @classmethod
    def no_duplicate_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("A section cites the same source more than once.")
        return value


class Exercise(BaseModel):
    task: NonEmptyStr
    hint: NonEmptyStr


class LessonGrounding(BaseModel):
    document_fingerprint: NonEmptyStr
    source_filename: NonEmptyStr
    retrieval_query: NonEmptyStr
    sources: list[SourceReference] = Field(min_length=1)


class LessonContent(BaseModel):
    lesson_number: int = Field(ge=1)
    title: NonEmptyStr
    introduction: NonEmptyStr
    sections: list[LessonSection] = Field(min_length=2, max_length=4)
    summary: NonEmptyStr
    exercises: list[Exercise] = Field(min_length=2, max_length=2)
    grounding: LessonGrounding | None = None

    @property
    def is_grounded(self) -> bool:
        return self.grounding is not None

    @property
    def cited_pages(self) -> list[int]:
        if self.grounding is None:
            return []
        return sorted({source.page_number for source in self.grounding.sources})

    @model_validator(mode="after")
    def check_citations(self, info: ValidationInfo):
        cited = [source_id for section in self.sections for source_id in section.source_ids]
        if self.grounding is None:
            if cited:
                raise ValueError("An ordinary lesson must not contain citations.")
            return self

        source_ids = [source.chunk_id for source in self.grounding.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("The same source is listed more than once.")
        if not cited:
            raise ValueError("A grounded lesson must cite at least one passage.")
        unknown = set(cited) - set(source_ids)
        if unknown:
            raise ValueError(f"Citations refer to unknown sources: {sorted(unknown)}")
        uncited = set(source_ids) - set(cited)
        if uncited:
            raise ValueError(f"Sources never cited by a section: {sorted(uncited)}")

        context = info.context or {}
        expected_fingerprint = context.get("expected_fingerprint")
        if expected_fingerprint and self.grounding.document_fingerprint != expected_fingerprint:
            raise ValueError("The sources do not belong to the currently loaded PDF.")
        allowed = context.get("allowed_sources")
        if allowed is not None:
            allowed_by_id = {source.chunk_id: source for source in allowed}
            for source in self.grounding.sources:
                if allowed_by_id.get(source.chunk_id) != source:
                    raise ValueError(f"Source {source.chunk_id} does not match the retrieved passage.")
        return self


def validate_lesson(
    data: dict,
    expected_lesson_number: int | None = None,
    expected_title: str | None = None,
    allowed_sources: list[SourceReference] | None = None,
    expected_fingerprint: str | None = None,
) -> LessonContent:
    context = {}
    if allowed_sources is not None:
        context["allowed_sources"] = allowed_sources
    if expected_fingerprint is not None:
        context["expected_fingerprint"] = expected_fingerprint
    lesson = LessonContent.model_validate(data, context=context)
    if expected_lesson_number is not None and lesson.lesson_number != expected_lesson_number:
        raise ValueError(f"Expected lesson number {expected_lesson_number}, got {lesson.lesson_number}.")
    if expected_title is not None and lesson.title != expected_title.strip():
        raise ValueError("The lesson title does not match the outline.")
    return lesson
