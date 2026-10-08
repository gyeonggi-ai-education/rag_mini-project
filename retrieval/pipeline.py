from retrieval.dense import dense_retrieve, dense_retrieve_many
from retrieval.multi_query import multi_query_retrieve
from retrieval.rerank import rerank

MODES = ("baseline", "rerank", "multi_query", "combined")
# 팀 공유 contracts.py의 모드 이름도 받아 준다.
MODE_ALIASES = {"dense": "baseline", "multi_query_rerank": "combined"}


def normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def retrieve(question: str, mode: str = "baseline", candidate_k: int = 20, final_k: int = 5, stats: dict | None = None) -> list[dict]:
    """검색 모드에 따라 최종 근거 목록을 돌려준다. API와 평가 도구가 같은 함수를 쓴다.

    - baseline: Dense
    - rerank: Dense 후보 → LLM Rerank
    - multi_query: 질의 확장 → 질의별 Dense(임베딩 1회) → RRF 병합
    - combined: multi_query 병합 결과 → Rerank 1회
    stats를 넘기면 Rerank 전 후보를 stats["candidates"]에 담는다(평가용). multi_query 계열은 확장 질의 정보도 채운다.
    """
    mode = normalize_mode(mode)
    if mode not in MODES:
        raise ValueError(f"지원하지 않는 모드: {mode} (사용 가능: {MODES})")

    if mode in ("multi_query", "combined"):
        candidates = multi_query_retrieve(
            question, candidate_k=candidate_k, dense_many_fn=dense_retrieve_many, stats=stats
        )
    else:
        candidates = dense_retrieve(question, candidate_k=candidate_k)
    if stats is not None:
        stats["candidates"] = candidates

    if mode in ("rerank", "combined"):
        return rerank(question, candidates, final_k=final_k)
    return candidates[:final_k]
