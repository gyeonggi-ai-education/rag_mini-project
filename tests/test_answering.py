"""Synthetic evidence only; no external model or legal assertions."""

import copy
import json
import sys
from types import SimpleNamespace as NS

import pytest
import httpx
from openai import APITimeoutError
from pydantic import ValidationError

from common.answering import (
    AnsweringError, AnsweringTimeout, CitationValidationError, GroundedAnswerer,
)
from common.contracts import AnswerResponse, Chunk, SearchHit, SourceLocation


def hit(cid="a", **updates):
    chunk = Chunk(**(dict(
        chunk_id=cid, document_id="fixture", document_version="v1",
        source_uri="fixture://synthetic.pdf", content="SYNTHETIC alpha.\n beta.",
        page_start=2, page_end=3, article=None, paragraph=None,
    ) | updates))
    return SearchHit(chunk=chunk, rank=1, retrieval_score=0.8)


def output(status="answered"):
    data = dict(status=status, summary="Synthetic summary", claims=[], citations=[])
    if status in {"answered", "partial"}:
        data.update(claims=[dict(text="Synthetic claim", citation_ids=["c1"])],
                    citations=[dict(
                        **{f: getattr(hit().chunk, f) for f in SourceLocation.model_fields},
                        citation_id="c1", excerpt="alpha.",
                    )])
    if status == "partial":
        data["limitations"] = ["Synthetic unknown condition"]
    if status == "needs_clarification":
        data["clarification_question"] = "Synthetic condition?"
    return data


class Model:
    def __init__(self, data=None, error=None):
        self.data = output() if data is None else data
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return NS(content=json.dumps(self.data, ensure_ascii=False))


@pytest.mark.parametrize("status", [
    "answered", "partial", "insufficient_evidence", "needs_clarification",
])
def test_four_states_and_only_used_evidence(status):
    model = Model(output(status))
    result = GroundedAnswerer(model=model).answer("  synthetic question  ", [hit(), hit("b")])
    assert isinstance(result, AnswerResponse)
    assert result.status == status
    assert [c.chunk_id for c in result.citations] == (["a"] if result.claims else [])
    result.validate_context([hit().chunk, hit("b").chunk])
    payload = json.loads(model.calls[0][1][1])
    assert payload["question"] == "synthetic question"
    assert [c["citation_id"] for c in payload["context"]] == ["c1", "c2"]
    assert payload["context"][0]["content"] == hit().chunk.content
    assert "retrieval_score" not in payload["context"][0]


@pytest.mark.parametrize("field,value", [
    ("citation_id", "absent"), ("chunk_id", "missing"),
    ("document_version", "v2"), ("document_id", "other"),
    ("source_uri", "fixture://other"), ("page_start", 1),
    ("page_end", 4), ("article", "invented"), ("paragraph", "invented"),
    ("item", "invented"), ("printed_page_start", 1),
    ("parent_article_id", "invented"), ("is_supplementary", True),
    ("excerpt", "not in original"), ("excerpt", "alpha. beta."),
])
def test_reject_fabricated_citation(field, value):
    data = output()
    data["citations"][0][field] = value
    if field == "citation_id":
        data["claims"][0]["citation_ids"] = [value]
    with pytest.raises(CitationValidationError):
        GroundedAnswerer(model=Model(data)).answer("question", [hit()])


def test_context_id_cannot_be_rebound_to_another_valid_chunk():
    data = output()
    data["citations"][0]["chunk_id"] = "b"
    with pytest.raises(CitationValidationError):
        GroundedAnswerer(model=Model(data)).answer("question", [hit(), hit("b")])


@pytest.mark.parametrize("change", ["missing", "unused", "duplicate", "uncited", "bad_state"])
def test_reject_invalid_claims_and_state(change):
    data = output()
    if change == "missing":
        data["claims"][0]["citation_ids"] = ["absent"]
    elif change == "unused":
        data["citations"].append(data["citations"][0] | {"citation_id": "c2"})
    elif change == "duplicate":
        data["citations"] *= 2
    elif change == "uncited":
        data["claims"][0]["citation_ids"] = []
    else:
        data["status"] = "partial"
    with pytest.raises(CitationValidationError):
        GroundedAnswerer(model=Model(data)).answer("question", [hit()])


def test_empty_context_does_not_initialize_or_call_model(monkeypatch):
    def forbidden():
        pytest.fail("empty evidence must not initialize model")
    monkeypatch.setitem(sys.modules, "common.ai_model", NS(get_llm_model=forbidden))
    result = GroundedAnswerer().answer("question", [])
    assert result.status == "insufficient_evidence"
    assert result.claims == result.citations == []


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001, None])
def test_invalid_input_before_model(question):
    model = Model()
    with pytest.raises(ValidationError):
        GroundedAnswerer(model=model).answer(question, [hit()])
    assert model.calls == []


@pytest.mark.parametrize("hits", [[hit(), hit()], ["invalid"], None])
def test_invalid_context_is_system_error(hits):
    model = Model()
    with pytest.raises(AnsweringError):
        GroundedAnswerer(model=model).answer("question", hits)
    assert model.calls == []


@pytest.mark.parametrize("error,kind", [
    (RuntimeError("synthetic-sensitive-detail"), AnsweringError),
    (TimeoutError("synthetic-sensitive-detail"), AnsweringTimeout),
    (httpx.ReadTimeout("synthetic-sensitive-detail"), AnsweringTimeout),
    (APITimeoutError(request=httpx.Request("POST", "https://fixture.invalid")), AnsweringTimeout),
])
def test_dependency_failure_is_not_insufficient_evidence(error, kind, caplog):
    with pytest.raises(kind) as caught:
        GroundedAnswerer(model=Model(error=error)).answer("question", [hit()])
    assert "synthetic-sensitive-detail" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert not caplog.records


@pytest.mark.parametrize("raw", ["not JSON", "{}", "[]", "null", "```json\n{}\n```", 123])
def test_invalid_model_output_is_validation_failure(raw):
    model = NS(invoke=lambda messages: NS(content=raw))
    with pytest.raises(CitationValidationError):
        GroundedAnswerer(model=model).answer("question", [hit()])


def test_default_factory_and_independent_requests(monkeypatch):
    model = Model()
    calls = []
    def factory():
        calls.append(True)
        return model
    monkeypatch.setitem(sys.modules, "common.ai_model", NS(get_llm_model=factory))
    answerer = GroundedAnswerer()
    answerer.answer("first", [hit()])
    answerer.answer("second", [hit()])
    assert calls == [True]
    assert len(model.calls[1]) == 2
    assert json.loads(model.calls[1][1][1])["question"] == "second"
    assert "first" not in model.calls[1][1][1]


def test_initialization_failure_is_sanitized(monkeypatch):
    def factory():
        raise RuntimeError("synthetic-sensitive-detail")
    monkeypatch.setitem(sys.modules, "common.ai_model", NS(get_llm_model=factory))
    with pytest.raises(AnsweringError, match="initialization failed"):
        GroundedAnswerer().answer("question", [hit()])


def test_structure_validation_does_not_claim_semantic_accuracy():
    data = copy.deepcopy(output())
    data["claims"][0]["text"] = "Synthetic unsupported meaning"
    # Deliberately passes structural checks; semantic QA remains a separate task.
    result = GroundedAnswerer(model=Model(data)).answer("question", [hit()])
    assert result.claims[0].text == data["claims"][0]["text"]
