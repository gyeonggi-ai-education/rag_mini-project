"""Network-free HTTP regression tests using synthetic document evidence."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from httpx import ReadTimeout

from app.main import AskService, app, create_app, get_ask_service
from common.answering import AnsweringError, AnsweringTimeout, CitationValidationError, GroundedAnswerer
from common.contracts import AnswerResponse, Chunk, SearchHit
from common.retrieval import DenseRetriever, RetrievalError


QUESTION = "테스트 문서의 조건은 무엇인가요?"
SECRET = "synthetic-private-marker"


def evidence():
    chunk = Chunk(chunk_id="chunk-1", document_id="synthetic", document_version="v1",
                  source_uri="fixture.pdf", article="테스트 조", paragraph="테스트 항",
                  page_start=2, page_end=3, content="합성 테스트 조건과 예외입니다.")
    return SearchHit(chunk=chunk, rank=1, retrieval_score=0.8)


def output(status):
    result = {"status": status, "summary": "합성 테스트 설명"}
    if status in {"answered", "partial"}:
        source = evidence().chunk.model_dump(exclude={"content"})
        result.update(claims=[{"text": "합성 테스트 주장", "citation_ids": ["c1"]}],
                      citations=[source | {"citation_id": "c1", "excerpt": "테스트 조건"}])
    if status == "partial":
        result["limitations"] = ["나머지는 확인 불가"]
    if status == "needs_clarification":
        result["clarification_question"] = "대상을 알려 주세요."
    return result


class Retriever:
    def __init__(self, *, hits=None, error=None):
        self.hits = [evidence()] if hits is None else hits
        self.error = error
        self.calls = []

    def retrieve(self, question, mode, k):
        self.calls.append((question, mode, k))
        if self.error:
            raise self.error
        return self.hits


class Model:
    def __init__(self, *, status="answered", error=None, payload=None):
        self.status, self.error, self.payload = status, error, payload
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return SimpleNamespace(content=json.dumps(
            output(self.status) if self.payload is None else self.payload, ensure_ascii=False))


def setup_client(*, status="answered", hits=None, retrieval_error=None, model_error=None, payload=None):
    retriever = Retriever(hits=hits, error=retrieval_error)
    model = Model(status=status, error=model_error, payload=payload)
    service = AskService(retriever=retriever, answerer=GroundedAnswerer(model=model), k=3)
    return TestClient(create_app(service=service)), retriever, model


def test_preserve_root():
    assert TestClient(app).get("/").json() == {"message": "RAG API"}


@pytest.mark.parametrize("status", ["answered", "partial", "insufficient_evidence", "needs_clarification"])
def test_all_result_states_and_provenance(status):
    client, retriever, model = setup_client(status=status)
    response = client.post("/ask", json={"question": "  " + QUESTION + "  "})
    assert response.status_code == 200
    result = AnswerResponse.model_validate(response.json())
    assert result.status == status
    assert retriever.calls == [(QUESTION, "dense", 3)]
    assert len(model.calls) == 1
    if result.citations:
        result.validate_context([evidence().chunk])
        assert result.citations[0].page_end == 3


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": " \n\t"}, {"question": "가" * 2001},
    {"question": None}, {"question": 12}, {"question": []},
    {"question": QUESTION, "search_mode": "invalid"},
    {"question": QUESTION, "search_mode": None},
    {"question": QUESTION, "unknown": SECRET},
])
def test_invalid_input_never_calls_dependencies(payload):
    client, retriever, model = setup_client()
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    assert retriever.calls == model.calls == []
    assert SECRET not in response.text


def test_malformed_json_is_sanitized():
    client, retriever, model = setup_client()
    response = client.post("/ask", content='{"question": "' + SECRET,
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert SECRET not in response.text
    assert retriever.calls == model.calls == []


def test_question_length_boundary():
    client, retriever, _ = setup_client(hits=[])
    assert client.post("/ask", json={"question": "가" * 2000}).status_code == 200
    assert len(retriever.calls[0][0]) == 2000


@pytest.mark.parametrize("mode", ["dense", "rerank", "multi_query", "multi_query_rerank"])
def test_contract_modes_forwarded_to_injected_retriever(mode):
    client, retriever, _ = setup_client(hits=[])
    assert client.post("/ask", json={"question": QUESTION, "search_mode": mode}).status_code == 200
    assert retriever.calls == [(QUESTION, mode, 3)]


def test_empty_search_does_not_invoke_model():
    client, _, model = setup_client(hits=[])
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["citations"] == []
    assert model.calls == []


@pytest.mark.parametrize("stage,error,code", [
    ("retrieval", RetrievalError(SECRET), 503),
    ("retrieval", RuntimeError(SECRET), 503),
    ("retrieval", ValueError(SECRET), 503),
    ("retrieval", TimeoutError(SECRET), 504),
    ("retrieval", ReadTimeout(SECRET), 504),
    ("answerer", AnsweringError(SECRET), 503),
    ("answerer", CitationValidationError(SECRET), 503),
    ("model", RuntimeError(SECRET), 503),
    ("answerer", AnsweringTimeout(SECRET), 504),
    ("model", TimeoutError(SECRET), 504),
    ("model", ReadTimeout(SECRET), 504),
])
def test_dependency_failures_are_sanitized(stage, error, code, caplog):
    caplog.set_level("INFO")
    client, retriever, model = setup_client(
        retrieval_error=error if stage == "retrieval" else None,
        model_error=error if stage == "model" else None)
    if stage == "answerer":
        class FailedAnswerer:
            def answer(self, question, hits):
                raise error

        client.app.state.ask_service.answerer = FailedAnswerer()
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == code
    assert SECRET not in response.text + caplog.text
    assert QUESTION not in caplog.text
    assert "Traceback" not in response.text
    if stage == "retrieval":
        assert model.calls == []


def test_invalid_citation_is_not_exposed():
    payload = output("answered")
    payload["citations"][0]["page_start"] = 1
    client, _, _ = setup_client(payload=payload)
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert "claims" not in response.json()


def test_mutated_answer_contract_is_revalidated():
    client, _, _ = setup_client()

    class InvalidAnswerer:
        def answer(self, question, hits):
            result = AnswerResponse.model_validate(output("answered"))
            result.citations[0].excerpt = SECRET
            return result

    client.app.state.ask_service.answerer = InvalidAnswerer()
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert SECRET not in response.text


@pytest.mark.parametrize("stage", ["collection", "embedding", "search"])
def test_dense_dependency_timeouts_return_504(stage):
    class Embedding:
        model = "synthetic-embedding"

        def embed_query(self, question):
            if stage == "embedding":
                raise ReadTimeout(SECRET)
            return [0.1, 0.2]

    class Store:
        def get_collection(self, **kwargs):
            if stage == "collection":
                raise TimeoutError(SECRET)
            return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
                vectors=SimpleNamespace(size=2, distance="Cosine"))))

        def query_points(self, **kwargs):
            raise ReadTimeout(SECRET)

    retriever = DenseRetriever(collection="synthetic-v1", document_id="synthetic",
                               document_version="v1", embedding_record={
                                   "provider": "monorouter", "model": "synthetic-embedding",
                                   "dimensions": 2}, embedding_model=Embedding(), qdrant_client=Store())
    model = Model()
    client = TestClient(create_app(service=AskService(
        retriever=retriever, answerer=GroundedAnswerer(model=model), k=1)))
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 504
    assert SECRET not in response.text
    assert model.calls == []


def test_requests_are_independent_and_not_logged(caplog):
    caplog.set_level("INFO")
    client, retriever, model = setup_client()
    for question in [QUESTION, "별도의 합성 질문"]:
        assert client.post("/ask", json={"question": question}).status_code == 200
        assert question not in caplog.text
    assert len(retriever.calls) == 2
    assert len(model.calls[1]) == 2
    assert QUESTION not in model.calls[1][1][1]


def test_unconfigured_service_and_override():
    application = create_app()
    client = TestClient(application)
    assert client.post("/ask", json={"question": QUESTION}).status_code == 503
    assert client.post("/ask", json={"question": ""}).status_code == 422
    application.dependency_overrides[get_ask_service] = lambda: AskService(
        retriever=Retriever(hits=[]), answerer=GroundedAnswerer(), k=1)
    assert client.post("/ask", json={"question": QUESTION}).status_code == 200


@pytest.mark.parametrize("k", [0, -1, True, 1.5, None])
def test_service_requires_explicit_positive_k(k):
    with pytest.raises(ValueError):
        AskService(retriever=Retriever(), answerer=GroundedAnswerer(), k=k)
