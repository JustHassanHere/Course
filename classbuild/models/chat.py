import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from classbuild.models.course import NonEmptyStr
from classbuild.models.lesson import SourceReference

CITATION_MARKER = re.compile(r"\[(p\d+-c\d+)\]")


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: NonEmptyStr
    mode: Literal["lesson", "pdf"] = "lesson"
    status: Literal["answered", "insufficient_evidence"] = "answered"
    sources: list[SourceReference] = Field(default_factory=list)
    document_fingerprint: str | None = None
    truncated: bool = False

    @property
    def cited_pages(self) -> list[int]:
        return sorted({source.page_number for source in self.sources})

    @model_validator(mode="after")
    def check_sources(self):
        is_pdf_answer = self.role == "assistant" and self.mode == "pdf"
        if self.sources and not is_pdf_answer:
            raise ValueError("Only PDF-mode assistant answers may have sources.")
        if not is_pdf_answer:
            return self

        if self.sources and not self.document_fingerprint:
            raise ValueError("A cited answer must record the PDF fingerprint.")
        if self.status == "insufficient_evidence":
            if self.sources:
                raise ValueError("An insufficient-evidence reply must not cite anything.")
            return self

        if not self.sources:
            raise ValueError("An answered PDF reply must cite at least one passage.")
        ids = [source.chunk_id for source in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("A source is listed more than once.")
        unknown = set(CITATION_MARKER.findall(self.content)) - set(ids)
        if unknown:
            raise ValueError(f"The answer cites passages that are not its sources: {sorted(unknown)}")
        return self
