import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from evaluation.results_view import get_results, list_runs

router = APIRouter()

STATIC_DIR = Path(__file__).parent / "static"
STAMP_RE = re.compile(r"^\d{8}_\d{6}$")  # 실행 시각(YYYYMMDD_HHMMSS). 파일 검색 패턴에 들어가므로 형식을 제한한다.


@router.get("/eval", include_in_schema=False)
def eval_page():
    return FileResponse(STATIC_DIR / "eval.html")


@router.get("/api/eval/runs")
def runs():
    """평가 실행 목록(최신순). 모드별 결과는 evaluation/results/{stamp}_{mode}.json 에서 읽는다."""
    return {"runs": [r for r in list_runs() if r["compatible"]]}


@router.get("/api/eval/runs/{stamp}")
def run_detail(stamp: str):
    """한 번의 실행: 모드별 지표 + 질문별 모드 비교 (evaluation.results_view.get_results)."""
    if not STAMP_RE.match(stamp):
        raise HTTPException(status_code=400, detail="stamp 형식이 올바르지 않습니다. 예: 20261008_155314")
    try:
        data = get_results(stamp)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    for q in data["questions"]:  # 화면에서 쓰지 않는 큰 필드는 보내지 않는다
        for m in q["modes"].values():
            m.pop("per_query", None)
        q.pop("chunks", None)
    return data
