"""평가 결과 조회 API 라우터 (FastAPI에 붙이기용)

app/main.py 에서 두 줄만 추가하면 된다 (API 담당 B):
    from evaluation.api import router as eval_router
    app.include_router(eval_router)

- 저장된 결과 파일만 읽는다. 평가 실행(run_eval)·LLM은 불러오지 않는다.
- Qdrant는 원문을 요청한 엔드포인트(with_source=true, /eval/chunks)에서만 연결한다.
- 결과·문항이 없으면 404, 원본 저장소 연결 실패는 503으로 응답한다.
"""

from fastapi import APIRouter, HTTPException, Query

from evaluation.results_view import get_question, get_results, list_runs

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/runs")
def eval_runs():
    """평가 실행 목록 (최신순)"""
    return list_runs()


@router.get("/results")
def eval_results(stamp: str | None = None):
    """평가 결과 전체 (stamp 생략 시 가장 최근 실행)"""
    try:
        return get_results(stamp)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/results/questions/{question_id}")
def eval_question(question_id: str, stamp: str | None = None, with_source: bool = False):
    """문항 하나: 모드별 검색·답변과 청크별 역할. with_source=true면 청크 원문 전체 포함"""
    try:
        return get_question(stamp, question_id, with_source=with_source)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # Qdrant 연결 실패 등
        raise HTTPException(status_code=503, detail=f"원본 청크를 불러오지 못했습니다: {e}")


@router.get("/chunks")
def eval_chunks(ids: list[str] = Query(..., description="chunk_id 목록"), collection: str | None = None):
    """chunk_id로 원본 청크 원문 조회"""
    from evaluation.chunk_source import get_chunks

    try:
        return get_chunks(ids, collection=collection)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"원본 청크를 불러오지 못했습니다: {e}")
