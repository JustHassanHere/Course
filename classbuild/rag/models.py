from pydantic import BaseModel, Field, computed_field, model_validator


class PageText(BaseModel):
    page_number: int = Field(ge=1)
    text: str  # cleaned text; empty for image-only pages


class DocumentChunk(BaseModel):
    chunk_id: str
    source_filename: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)  # position within the page, starting at 1 (p3-c2 -> 2)
    text: str = Field(min_length=1)

    @property
    def char_count(self) -> int:
        return len(self.text)


class ProcessedDocument(BaseModel):
    source_filename: str
    fingerprint: str
    page_count: int = Field(ge=1)
    text_page_count: int = Field(ge=0)
    empty_pages: list[int] = Field(default_factory=list)
    word_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    chunk_size: int
    chunk_overlap: int
    chunks: list[DocumentChunk]

    @computed_field
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.text_page_count + len(self.empty_pages) != self.page_count:
            raise ValueError("Page counts do not add up.")
        ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Chunk ids must be unique.")
        return self

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
