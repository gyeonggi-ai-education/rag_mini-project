"""All fixtures are synthetic markers, not legal text or legal evidence."""

import pytest
from pydantic import ValidationError

from common.contracts import (
    MAX_QUESTION_LENGTH, AnswerResponse, AskRequest, Chunk, Citation,
    Claim, Document, SearchHit,
)


@pytest.fixture
def chunk_data():
    return dict(
        chunk_id="synthetic-chunk", document_id="synthetic-document",
        document_version="fixture-v1", source_uri="fixture://not-a-law.pdf",
        content="SYNTHETIC ONLY: alpha marker. beta marker.",
        article=None, paragraph=None, item=None, page_start=1, page_end=2,
    )


@pytest.fixture
def citation_data(chunk_data):
    return {k: v for k, v in chunk_data.items() if k != "content"} | {
        "citation_id": "c1", "excerpt": "alpha marker.",
    }


@pytest.mark.parametrize("question", ["", " ", "\n\t", "x" * 2001, None, 123])
def test_invalid_question(question):
    with pytest.raises(ValidationError):
        AskRequest(question=question)


def test_question_limit_and_default():
    assert MAX_QUESTION_LENGTH == 2000
    request = AskRequest(question="가" * MAX_QUESTION_LENGTH)
    assert request.search_mode == "dense"
    assert AskRequest(question="  synthetic question  ").question == "synthetic question"


@pytest.mark.parametrize("mode", ["dense", "rerank", "multi_query", "multi_query_rerank"])
def test_modes(mode):
    assert AskRequest(question="synthetic question", search_mode=mode).search_mode == mode


@pytest.mark.parametrize("mode", ["unknown", "", None])
def test_invalid_mode(mode):
    with pytest.raises(ValidationError):
        AskRequest(question="synthetic question", search_mode=mode)


def test_document():
    document = Document(
        document_id="synthetic-document", document_version="fixture-v1",
        title="SYNTHETIC, NOT A LAW", source_uri="fixture://not-a-law.pdf",
        pdf_sha256="a" * 64, processed_scope="partial",
    )
    assert Document.model_validate_json(document.model_dump_json()) == document
    with pytest.raises(ValidationError):
        Document(**(document.model_dump() | {"pdf_sha256": "invalid"}))
    with pytest.raises(ValidationError):
        Document(**(document.model_dump() | {"processed_scope": "unknown"}))


@pytest.mark.parametrize("field", ["document_id", "document_version", "source_uri"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_source(chunk_data, field, value):
    with pytest.raises(ValidationError):
        Chunk(**(chunk_data | {field: value}))


def test_omitted_source(chunk_data):
    del chunk_data["source_uri"]
    with pytest.raises(ValidationError):
        Chunk(**chunk_data)


@pytest.mark.parametrize("updates", [
    {"page_start": 0}, {"page_start": 3}, {"page_end": None},
    {"page_start": True}, {"page_end": 1.5}, {"content": " "},
    {"article": " "}, {"printed_page_start": 3, "printed_page_end": 2},
    {"printed_page_start": 1},
])
def test_invalid_chunk(chunk_data, updates):
    with pytest.raises(ValidationError):
        Chunk(**(chunk_data | updates))


def test_chunk_preserves_original_text_and_optional_structure(chunk_data):
    chunk_data["content"] = "  SYNTHETIC ONLY\n"
    chunk = Chunk(**chunk_data)
    assert chunk.content == chunk_data["content"]
    assert chunk.article is None and chunk.paragraph is None
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_search_hit(chunk_data):
    hit = SearchHit(chunk=Chunk(**chunk_data), rank=1, retrieval_score=0.3, rerank_score=0.7)
    assert hit.retrieval_score != hit.rerank_score
    for updates in [{"rank": 0}, {"retrieval_score": float("nan")}, {"rerank_score": float("inf")}]:
        with pytest.raises(ValidationError):
            SearchHit(**(hit.model_dump() | updates))


def response(citation_data, **updates):
    return AnswerResponse(**(dict(
        status="answered", summary="Synthetic explanation",
        claims=[Claim(text="Synthetic claim", citation_ids=["c1"])],
        citations=[Citation(**citation_data)],
    ) | updates))


def test_grounded_response_roundtrip_and_context(chunk_data, citation_data):
    result = response(citation_data)
    result.validate_context([Chunk(**chunk_data)])
    assert AnswerResponse.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("updates", [
    {"claims": [dict(text="Synthetic claim", citation_ids=["missing"])]},
    {"claims": [dict(text="Synthetic claim", citation_ids=[])]},
    {"claims": []}, {"citations": []},
])
def test_response_requires_exact_used_citations(citation_data, updates):
    with pytest.raises(ValidationError):
        response(citation_data, **updates)


def test_duplicate_and_unused_citations(citation_data):
    c = Citation(**citation_data)
    for citations in [[c, c], [c, Citation(**(citation_data | {"citation_id": "unused"}))]]:
        with pytest.raises(ValidationError):
            response(citation_data, citations=citations)
    with pytest.raises(ValidationError):
        Claim(text="Synthetic claim", citation_ids=["c1", "c1"])


@pytest.mark.parametrize("field,value", [
    ("document_version", "other"), ("document_id", "other"),
    ("source_uri", "fixture://other"), ("page_end", 3),
    ("article", "invented-marker"), ("excerpt", "fabricated marker"),
    ("chunk_id", "absent"),
])
def test_context_rejects_mismatched_evidence(chunk_data, citation_data, field, value):
    result = response(citation_data | {field: value})
    with pytest.raises(ValueError):
        result.validate_context([Chunk(**chunk_data)])


def test_context_rejects_duplicate_chunk_identity(chunk_data, citation_data):
    chunk = Chunk(**chunk_data)
    with pytest.raises(ValueError):
        response(citation_data).validate_context([chunk, chunk])


def test_four_result_states(citation_data):
    assert response(citation_data).status == "answered"
    assert response(citation_data, status="partial", limitations=["Synthetic unknown"]).status == "partial"
    for status in ["insufficient_evidence", "needs_clarification"]:
        result = AnswerResponse(
            status=status, summary="Synthetic outcome",
            clarification_question="Synthetic condition?" if status == "needs_clarification" else None,
        )
        assert result.claims == [] and result.citations == []


@pytest.mark.parametrize("updates", [
    {"status": "unknown"}, {"status": "partial"},
    {"status": "insufficient_evidence"}, {"status": "needs_clarification"},
    {"clarification_question": "Synthetic question?"},
])
def test_inconsistent_states(citation_data, updates):
    with pytest.raises(ValidationError):
        response(citation_data, **updates)


def test_clarification_requires_question():
    with pytest.raises(ValidationError):
        AnswerResponse(status="needs_clarification", summary="Synthetic outcome")


def test_contract_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AskRequest(question="synthetic question", unsupported=True)
