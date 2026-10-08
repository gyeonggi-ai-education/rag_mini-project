"""답변 평가 (project-plan.md 7장: 정답성, 관련성, 근거 충실성, 근거 없는 주장, 답변 불가 처리)

API 호출을 줄이기 위해 두 단계로 나눈다.
- 규칙 기반 채점 (기본, API 호출 없음): 필수 답변 요소 포함률, 답변 문장이 근거 원문과 겹치는 비율
- LLM 채점 (선택, --judge llm): 한 문항의 여러 모드 답변을 한 번의 호출로 함께 채점

평가 기준은 실험 전에 고정한다. 기준을 바꾸면 버전을 올린다.
두 채점 모두 보조 지표이며, 리포트의 답변·근거 원문을 사람이 대조해 확인한다.
"""

import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

RULE_JUDGE_VERSION = "rule-v1"
JUDGE_VERSION = "llm-v3"  # v3: 문항 단위로 여러 답변을 한 번에 채점

KEY_POINT_THRESHOLD = 0.5  # 필수 요소의 글자 2-gram 중 절반 이상이 답변에 있으면 포함으로 본다
GROUNDED_THRESHOLD = 0.5  # 답변 문장의 2-gram 중 절반 이상이 근거 원문에 있으면 근거 있음으로 본다


# ---------- 규칙 기반 채점 (API 호출 없음) ----------

def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"[\s\"'“”‘’「」『』()·ㆍ,.]", "", text)
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def _overlap(part: str, whole: set[str]) -> float:
    grams = _bigrams(part)
    return len(grams & whole) / len(grams) if grams else 0.0


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(_bigrams(s)) >= 5]


def _refusal_result(item: dict, answer: dict, version: str) -> dict | None:
    """답변 불가 문항, 또는 답변 가능 문항을 거절한 경우 (채점 없이 판정)"""
    if not item["answerable"]:
        return {"judge_version": version, "refused_correctly": not answer["is_answerable"]}
    if not answer["is_answerable"]:
        return {"judge_version": version, "false_refusal": True}
    return None


def rule_judge(item: dict, answer: dict, used_chunks: list[dict]) -> dict:
    """글자 2-gram 겹침으로 필수 요소 포함률과 근거 겹침 비율을 계산 (근사치)"""
    refusal = _refusal_result(item, answer, RULE_JUDGE_VERSION)
    if refusal:
        return refusal

    answer_grams = _bigrams(answer["answer"])
    key_points = item.get("key_points", [])
    covered = [p for p in key_points if _overlap(p, answer_grams) >= KEY_POINT_THRESHOLD]

    evidence_grams = _bigrams(" ".join(c["content"] for c in used_chunks))
    sentences = _sentences(answer["answer"])
    ungrounded = [s for s in sentences if _overlap(s, evidence_grams) < GROUNDED_THRESHOLD]
    return {
        "judge_version": RULE_JUDGE_VERSION,
        "key_point_coverage": round(len(covered) / len(key_points), 3) if key_points else None,
        "missed_key_points": [p for p in key_points if p not in covered],
        "grounded_ratio": round(1 - len(ungrounded) / len(sentences), 3) if sentences else None,
        "ungrounded_sentences": ungrounded,
    }


# ---------- LLM 채점 (문항당 1회) ----------

RUBRIC = """채점 기준 (각 1~5점, 5가 가장 좋음)
- correctness(정답성): 필수 답변 요소(key_points)를 얼마나 정확히 포함하는가
- relevance(관련성): 질문에 직접 답하는가
- faithfulness(근거 충실성): 답변 내용이 그 답변의 '사용한 근거' 원문으로 뒷받침되는가
- unsupported_claims(근거 없는 주장): 사용한 근거 원문에 없는 내용을 사실처럼 말했다면 그 문장들을 나열
여러 답변이 번호와 함께 주어진다. 각 답변은 자기 근거만으로 독립적으로 채점하고, 같은 번호를 붙인다."""


class JudgedAnswer(BaseModel):
    id: int = Field(description="답변 번호")
    correctness: int = Field(ge=1, le=5, description="정답성 1~5")
    relevance: int = Field(ge=1, le=5, description="관련성 1~5")
    faithfulness: int = Field(ge=1, le=5, description="근거 충실성 1~5")
    unsupported_claims: list[str] = Field(description="근거 원문에 없는 주장 목록, 없으면 빈 목록")
    reason: str = Field(description="채점 이유 한두 문장")


class BatchJudgement(BaseModel):
    items: list[JudgedAnswer]


PROMPT = ChatPromptTemplate.from_messages([
    ("system", "당신은 법률 QA 답변을 엄격하게 채점하는 평가자다.\n" + RUBRIC),
    ("human", "질문: {question}\n\n필수 답변 요소(key_points):\n{key_points}\n\n{answers}"),
])

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        from common.ai_model import get_llm_model

        _chain = PROMPT | get_llm_model().with_structured_output(BatchJudgement, method="function_calling")
    return _chain


def llm_judge_batch(item: dict, entries: list[tuple[dict, list[dict]]]) -> list[dict | None]:
    """한 문항의 여러 답변을 한 번의 LLM 호출로 채점 → entries 순서대로 결과 (빠진 답변은 None)

    entries: [(answer, used_chunks), ...] — 거절·답변 불가 문항은 넣지 않는다 (호출한 쪽에서 걸러냄)
    """
    if not entries:
        return []
    blocks = []
    for i, (answer, used_chunks) in enumerate(entries, start=1):
        evidence = "\n".join(f"  {c['article']}: {c['content']}" for c in used_chunks) or "  (사용한 근거 없음)"
        blocks.append(f"[답변 {i}]\n사용한 근거 원문:\n{evidence}\n답변:\n{answer['answer']}")
    result = _get_chain().invoke({
        "question": item["question"],
        "key_points": "\n".join(f"- {p}" for p in item.get("key_points", [])),
        "answers": "\n\n".join(blocks),
    })
    by_id = {j.id: j for j in result.items}
    return [
        {"judge_version": JUDGE_VERSION, **by_id[i].model_dump(exclude={"id"})} if i in by_id else None
        for i in range(1, len(entries) + 1)
    ]
