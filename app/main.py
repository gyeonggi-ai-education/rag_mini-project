import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.eval_api import router as eval_router
from app.viewer import router as viewer_router
from common.config import CANDIDATE_K, FINAL_K
from generation.answer import generate_answer
from retrieval.pipeline import MODES, normalize_mode, retrieve

# 검색 모드는 내부 설정으로만 바꾼다. 기본은 Multi Query + Rerank(combined). 예: RETRIEVAL_MODE=rerank
RETRIEVAL_MODE = normalize_mode(os.getenv("RETRIEVAL_MODE", "combined"))
if RETRIEVAL_MODE not in MODES:
    raise ValueError(f"RETRIEVAL_MODE는 {MODES} 중 하나여야 합니다: {RETRIEVAL_MODE}")

app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.include_router(viewer_router)
app.include_router(eval_router)


class AskRequest(BaseModel):
    question: str
    # 선택. 없으면 서버 기본 모드(RETRIEVAL_MODE)를 쓴다. 기존 {"question": ...} 요청은 그대로 동작한다.
    mode: str | None = None


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/chat")


@app.get("/chat", include_in_schema=False)
def chat_page():
    return FileResponse(Path(__file__).parent / "static" / "chat.html")


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question이 비어 있습니다.")

    mode = normalize_mode(req.mode) if req.mode else RETRIEVAL_MODE
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 mode입니다: {req.mode} (사용 가능: {', '.join(MODES)})")

    try:
        chunks = retrieve(question, mode=mode, candidate_k=CANDIDATE_K, final_k=FINAL_K)
        result = generate_answer(question, chunks)
        # is_answerable·used_chunk_ids는 평가용 내부 필드라 /ask 응답에는 노출하지 않는다.
        return {"answer": result["answer"], "sources": result["sources"], "mode": mode}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"모델 또는 검색 호출에 실패했습니다: {type(e).__name__}")
