import json
import logging
import re

from common.ai_model import get_llm_model

SYSTEM_PROMPT = """너는 법률 검색 결과의 관련도를 평가하는 평가자다.
질문에 답하는 데 각 [후보]가 얼마나 도움이 되는지 0~10점으로 매긴다.
- 10: 질문에 직접 답하는 핵심 근거, 0: 전혀 관련 없음
- 후보의 내용만 보고 판단하고, 후보 번호마다 하나씩 점수를 매긴다.
반드시 아래 JSON 형식으로만 출력한다.
{"scores": [{"id": 후보 번호, "score": 점수}, ...]}"""

logger = logging.getLogger(__name__)

_llm = get_llm_model(max_tokens=1024)


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0))


def rerank(question: str, candidates: list[dict], final_k: int = 5) -> list[dict]:
    """LLM 기반 재정렬: 후보 전체를 한 번의 호출로 평가해 rerank_score를 채우고 상위 final_k를 돌려준다.

    retrieval_score는 그대로 두고, rank만 재정렬 결과 기준으로 다시 매긴다.
    """
    if not candidates:
        return []

    listing = "\n\n".join(
        f"[후보 {i}] {c.get('article', '')} {c.get('article_title') or c.get('title', '')}\n{c['content']}"
        for i, c in enumerate(candidates, start=1)
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"[질문]\n{question}\n\n{listing}"),
    ]
    try:
        scores = {
            s["id"]: float(s["score"])
            for s in _parse(_llm.invoke(messages).content)["scores"]
            if isinstance(s.get("id"), int)
        }
    except Exception as e:
        # 호출·파싱 실패 시 Dense 순서 그대로 반환한다. rerank_score는 None, 원인은 rerank_error(예외 클래스 이름)에 남긴다.
        logger.exception("rerank 실패: Dense 순서로 대체")
        return [
            {**c, "rank": rank, "rerank_error": type(e).__name__}
            for rank, c in enumerate(candidates[:final_k], start=1)
        ]

    # 점수가 없는 후보는 최하점 처리. 동점이면 기존 Dense 순서를 유지한다(sorted는 안정 정렬).
    scored = [
        {**c, "rerank_score": scores.get(i, 0.0)}
        for i, c in enumerate(candidates, start=1)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return [{**c, "rank": rank} for rank, c in enumerate(scored[:final_k], start=1)]
