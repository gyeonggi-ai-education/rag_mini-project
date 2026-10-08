"""HTTP adapter for explicitly configured, document-scoped QA services.

Inject AskService via create_app(service=...) or get_ask_service overrides.
The default app serves GET / and returns 503 for valid /ask requests until
configured. Collection, document version, embedding record and K are never
inferred here; configure a DenseRetriever and GroundedAnswerer explicitly.
"""

from collections.abc import Sequence
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from httpx import TimeoutException
from openai import APITimeoutError

from common.answering import AnsweringError, AnsweringTimeout
from common.contracts import AnswerResponse, AskRequest, SearchHit, SearchMode
from common.retrieval import RetrievalError


def _is_timeout(error: Exception) -> bool:
    timeout_types = (AnsweringTimeout, TimeoutError, TimeoutException, APITimeoutError)
    if isinstance(error, timeout_types):
        return True
    if not isinstance(error, (RetrievalError, AnsweringError)):
        return False
    # Existing adapters hide provider details with `raise ... from None`.
    # __context__ still retains the typed original failure for classification;
    # none of its text or traceback is returned or logged.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, timeout_types):
            return True
        error = error.__cause__ or error.__context__
    return False


class Retriever(Protocol):
    def retrieve(self, question: str, mode: SearchMode, k: int) -> Sequence[SearchHit]: ...


class Answerer(Protocol):
    def answer(self, question: str, hits: Sequence[SearchHit]) -> AnswerResponse: ...


class AskService:
    """One independent retrieve -> grounded answer operation per request."""

    def __init__(self, *, retriever: Retriever, answerer: Answerer, k: int):
        if type(k) is not int or k < 1:
            raise ValueError("Expected explicit positive integer k")
        self.retriever, self.answerer, self.k = retriever, answerer, k

    def ask(self, request: AskRequest) -> AnswerResponse:
        hits = self.retriever.retrieve(request.question, request.search_mode, self.k)
        result = self.answerer.answer(request.question, hits)
        # Revalidate even mutated contract objects before HTTP serialization.
        checked = AnswerResponse.model_validate(result.model_dump())
        checked.validate_context([hit.chunk for hit in hits])
        return checked


def get_ask_service(request: Request) -> AskService | None:
    # Do not raise here: FastAPI resolves dependencies before body validation.
    # Invalid input must remain 422 even on an unconfigured deployment.
    return request.app.state.ask_service


def create_app(*, service: AskService | None = None) -> FastAPI:
    application = FastAPI()
    application.state.ask_service = service

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # Default validation errors include raw input and validator context.
        # Use fixed text so credentials, question text and exception details
        # cannot be reflected, including malformed JSON and extra fields.
        return JSONResponse(status_code=422, content={"detail": "요청 입력을 확인해 주세요."})

    @application.get("/")
    def root():
        return {"message": "RAG API"}

    @application.post("/ask", response_model=AnswerResponse)
    def ask(body: AskRequest, configured: Annotated[AskService | None, Depends(get_ask_service)]):
        if configured is None:
            raise HTTPException(status_code=503, detail="답변 서비스를 사용할 수 없습니다.")
        try:
            return configured.ask(body)
        except Exception as error:
            # Fail closed for dependency, citation and unexpected adapter errors.
            # Never serialize or log the underlying exception or request body.
            if _is_timeout(error):
                raise HTTPException(status_code=504, detail="답변 처리 시간이 초과되었습니다.") from None
            raise HTTPException(status_code=503, detail="일시적으로 답변하지 못했습니다.") from None

    return application


app = create_app()
