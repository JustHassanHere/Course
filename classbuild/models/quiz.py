from pydantic import BaseModel, Field, model_validator

from classbuild.models.course import NonEmptyStr


class QuizQuestion(BaseModel):
    question: NonEmptyStr
    options: list[NonEmptyStr] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)  # exactly one correct option
    explanation: NonEmptyStr

    @model_validator(mode="after")
    def options_must_be_distinct(self):
        lowered = [option.lower() for option in self.options]
        if len(set(lowered)) != len(lowered):
            raise ValueError("Answer options must be distinct.")
        return self


class Quiz(BaseModel):
    questions: list[QuizQuestion] = Field(min_length=5, max_length=5)
