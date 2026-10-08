from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from qdrant_client import models

from common.config import COLLECTION_NAME
from common.qdrant import get_qdrant_client

router = APIRouter()
_qdrant = get_qdrant_client()

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_COLLECTION = COLLECTION_NAME
SCAN_MATCH_CAP = 100
# 화면에 보여줄 이름. Qdrant 컬렉션 이름은 검색 코드가 쓰므로 바꾸지 않는다.
DISPLAY_NAMES = {DEFAULT_COLLECTION: "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법"}


def _normalize(value) -> str:
    """띄어쓰기와 대소문자 차이를 무시하고 비교하기 위해 공백을 모두 제거한다."""
    return "".join(str(value).split()).lower()


def _parse_point_id(raw: str):
    return int(raw) if raw.isdigit() else raw


def _build_filter(filter_text: str | None):
    """'key:value' 형태의 payload 필터를 Qdrant Filter로 바꾼다. 'id:123'은 point ID 조회."""
    filter_text = (filter_text or "").strip()
    if not filter_text:
        return None, None
    if ":" not in filter_text:
        return None, None
    key, value = (s.strip() for s in filter_text.split(":", 1))
    if not key or not value:
        return None, None
    if key == "id":
        return None, _parse_point_id(value)
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]), None


def _point_to_dict(point, score: float | None = None, vector_length: int | None = None):
    return {
        "id": point.id,
        "payload": point.payload,
        "score": score,
        "vector_length": vector_length,
    }


def _vector_length(collection: str) -> int | None:
    try:
        vectors = _qdrant.get_collection(collection).config.params.vectors
        return vectors.size if hasattr(vectors, "size") else None
    except Exception:
        return None


@router.get("/viewer", include_in_schema=False)
def viewer_page():
    return FileResponse(STATIC_DIR / "viewer.html")


@router.get("/api/collections")
def list_collections():
    try:
        names = [c.name for c in _qdrant.get_collections().collections]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Qdrant 연결 실패: {type(e).__name__}")
    return {"collections": names, "default": DEFAULT_COLLECTION}


@router.get("/api/collections/{collection}")
def collection_info(collection: str):
    try:
        info = _qdrant.get_collection(collection)
    except Exception:
        raise HTTPException(status_code=404, detail=f"컬렉션을 찾을 수 없습니다: {collection}")
    return {
        "name": collection,
        "display_name": DISPLAY_NAMES.get(collection, collection),
        "status": str(info.status),
        "points_count": info.points_count,
        "vector_length": _vector_length(collection),
    }


@router.get("/api/collections/{collection}/points")
def list_points(
    collection: str,
    limit: int = Query(10, ge=1, le=100),
    offset: str | None = None,
    filter: str | None = Query(None, description="payload 필터(key:value) 또는 point ID(id:123)"),
):
    """포인트를 페이지 단위로 조회한다. next_offset을 offset에 넣으면 다음 페이지."""
    qfilter, point_id = _build_filter(filter)
    vlen = _vector_length(collection)
    keyword = (filter or "").strip()
    try:
        if keyword and ":" not in keyword:
            # 키 없이 입력하면 payload 값에 대한 부분 일치 검색. 데이터가 적은 동안만 쓰는 전체 스캔이다.
            needle = _normalize(keyword)
            matched, next_page = [], None
            while True:
                batch, next_page = _qdrant.scroll(collection, limit=256, offset=next_page, with_payload=True)
                matched += [p for p in batch if any(needle in _normalize(v) for v in (p.payload or {}).values())]
                if next_page is None or len(matched) >= SCAN_MATCH_CAP:
                    break
            return {"points": [_point_to_dict(p, vector_length=vlen) for p in matched[:SCAN_MATCH_CAP]],
                    "next_offset": None}
        if point_id is not None:
            points = _qdrant.retrieve(collection, ids=[point_id], with_payload=True)
            next_offset = None
        else:
            points, next_offset = _qdrant.scroll(
                collection,
                scroll_filter=qfilter,
                limit=limit,
                offset=_parse_point_id(offset) if offset else None,
                with_payload=True,
                with_vectors=False,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"포인트 조회 실패: {type(e).__name__}")
    return {
        "points": [_point_to_dict(p, vector_length=vlen) for p in points],
        "next_offset": next_offset,
    }


@router.get("/api/collections/{collection}/points/{point_id}/similar")
def similar_points(collection: str, point_id: str, limit: int = Query(5, ge=1, le=50)):
    """Find Similar: 해당 포인트와 벡터가 가까운 포인트를 찾는다(자기 자신 제외)."""
    vlen = _vector_length(collection)
    try:
        hits = _qdrant.query_points(
            collection,
            query=_parse_point_id(point_id),
            limit=limit,
            with_payload=True,
        ).points
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"유사 포인트 조회 실패: {type(e).__name__}")
    return {"points": [_point_to_dict(h, score=h.score, vector_length=vlen) for h in hits]}
