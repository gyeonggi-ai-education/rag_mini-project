"""Local QA contracts; validation checks structure, not legal faithfulness.

Questions are stripped and limited to 2,000 Unicode characters. PDF pages are
one-based physical pages. Unknown article/paragraph/item values stay None.
No provider configuration or external client is loaded by this module.
"""

from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


MAX_QUESTION_LENGTH = 2000
SearchMode = Literal["dense", "rerank", "multi_query", "multi_query_rerank"]
AnswerStatus = Literal["answered", "partial", "insufficient_evidence", "needs_clarification"]
NonBlank = Annotated[str, StringConstraints(strict=True, pattern=r"\S")]
Page = Annotated[int, Field(strict=True, ge=1)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AskRequest(Contract):
    question: Annotated[str, StringConstraints(
        strict=True, strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_LENGTH,
    )]
    search_mode: SearchMode = "dense"


class Document(Contract):
    document_id: NonBlank
    document_version: NonBlank
    title: NonBlank
    source_uri: NonBlank
    pdf_sha256: Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-fA-F]{64}$")]
    processed_scope: Literal["full", "partial"]


class SourceLocation(Contract):
    """Explicit provenance; URI is preserved without inventing a PDF link."""

    chunk_id: NonBlank
    document_id: NonBlank
    document_version: NonBlank
    source_uri: NonBlank
    article: NonBlank | None = None
    paragraph: NonBlank | None = None
    item: NonBlank | None = None
    page_start: Page
    page_end: Page
    printed_page_start: Page | None = None
    printed_page_end: Page | None = None
    parent_article_id: NonBlank | None = None
    is_supplementary: bool | None = None

    @model_validator(mode="after")
    def validate_pages(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("PDF page range is reversed")
        start, end = self.printed_page_start, self.printed_page_end
        if (start is None) != (end is None):
            raise ValueError("printed page range requires both endpoints")
        if start is not None and end is not None and end < start:
            raise ValueError("printed page range is reversed")
        return self


class Chunk(SourceLocation):
    # Preserve whitespace: excerpt checks must use the original content.
    content: NonBlank


class SearchHit(Contract):
    chunk: Chunk
    rank: Page
    retrieval_score: float
    rerank_score: float | None = None


class Citation(SourceLocation):
    citation_id: NonBlank
    excerpt: NonBlank


class Claim(Contract):
    text: NonBlank
    citation_ids: list[NonBlank] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        if len(self.citation_ids) != len(set(self.citation_ids)):
            raise ValueError("claim citation IDs must be unique")
        return self


class AnswerResponse(Contract):
    status: AnswerStatus
    summary: NonBlank
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[NonBlank] = Field(default_factory=list)
    clarification_question: NonBlank | None = None
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        ids = [citation.citation_id for citation in self.citations]
        if len(ids) != len(set(ids)):
            raise ValueError("response citation IDs must be unique")
        used = {cid for claim in self.claims for cid in claim.citation_ids}
        if used != set(ids):
            raise ValueError("citations must exactly match the IDs used by claims")
        if self.status in {"answered", "partial"}:
            if not self.claims:
                raise ValueError("grounded responses require cited claims")
            if self.status == "partial" and not self.limitations:
                raise ValueError("partial responses require explicit limitations")
        elif self.claims or self.citations:
            raise ValueError("unanswered responses cannot contain grounded claims")
        if self.status == "needs_clarification":
            if self.clarification_question is None:
                raise ValueError("clarification responses require a question")
        elif self.clarification_question is not None:
            raise ValueError("clarification question requires needs_clarification status")
        return self

    def validate_context(self, chunks: Sequence[Chunk]) -> None:
        """Check provenance and verbatim excerpts against supplied context.

        Call before exposing a generated answer. This cannot prove that a
        claim follows semantically from the excerpt; that needs separate QA.
        """
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(by_id) != len(chunks):
            raise ValueError("context chunk IDs must be unique")
        for citation in self.citations:
            chunk = by_id.get(citation.chunk_id)
            if chunk is None:
                raise ValueError("citation chunk is absent from context")
            for field in SourceLocation.model_fields:
                if getattr(citation, field) != getattr(chunk, field):
                    raise ValueError("citation provenance differs from context")
            if citation.excerpt not in chunk.content:
                raise ValueError("citation excerpt is absent from original content")
