"""Multi Query 검색 (C 담당)

원질문 + 확장 질의 → 질의별 Dense 검색 → chunk_id 기준 중복 제거 + RRF 병합

B의 dense_retrieve와 공통 모델 어댑터가 완성되기 전에도 개발·테스트할 수 있도록
dense_fn, llm을 인자로 주입받는다. 생략하면 공통 모듈을 사용한다.
"""

import logging
from collections.abc import Callable

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 초기 실험값 (project-plan.md 6장, 확정값 아님)
MAX_EXPANDED_QUERIES = 3
CANDIDATE_K = 20
RRF_K = 60

# 청크 dict: payload 필드 + retrieval_score, rerank_score, rank (형식은 C의 schemas/chunk.py 참고).
# 이 파일만 복사해도 동작하도록 별도 스키마 모듈에 의존하지 않는다.
RetrievedChunk = dict
DenseFn = Callable[[str, int], list[RetrievedChunk]]
# 질의 여러 개 → 질의별 Dense 결과 목록 (임베딩을 한 번의 요청으로 묶을 때)
DenseManyFn = Callable[[list[str], int], list[list[RetrievedChunk]]]


class ExpandedQueries(BaseModel):
    queries: list[str] = Field(description="원질문과 같은 의미를 다른 표현으로 바꾼 검색용 질의 목록")


# v1: 같은 의미의 다른 표현만 요청 → 일상어가 그대로 남아 법률 용어로 바뀌지 않음 (D04 실패)
# v2: 질의마다 역할을 나눠 법령 조문 용어·정의 조항 형태로 바꿔 쓰도록 요청
EXPAND_PROMPT_VERSION = "v2"

EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """당신은 한국 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」 조문 검색을 돕는다.
        사용자는 일상어로 묻지만 조문은 법률 용어로 쓰여 있다. 검색이 조문과 잘 맞도록 질의를 최대 {n}개 만든다.

        질의마다 역할을 다르게 한다.
        1. 일상어 표현을 법령에서 쓰일 정식 용어로 바꾼 질의 (예: 'AI' → '인공지능', '알려야' → '고지', '벌금' → '과태료 또는 벌칙')
        2. 질문의 상황이 법에서 어떤 개념·분류에 해당하는지 묻는 질의 (정의 조항은 '~이란 ~을 말한다' 형태로 쓰인다)
        3. 관련된 의무·요건·절차·제재를 조문 문체로 쓴 질의

        질문에 없는 사실이나 조건을 추가하지 않는다. 조문 번호를 추측해서 넣지 않는다."""
    ),
    ("human", "질문: {question}"),
])


class ExpandedItem(BaseModel):
    id: int = Field(description="입력 질문 번호")
    queries: list[str] = Field(description="해당 질문의 검색용 질의 목록")


class BatchExpandedQueries(BaseModel):
    items: list[ExpandedItem]


# 여러 질문을 한 번의 요청으로 확장 (API 호출 수 절감). 규칙은 EXPAND_PROMPT와 같다.
EXPAND_BATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EXPAND_PROMPT.messages[0].prompt.template
     + "\n\n        여러 질문이 번호와 함께 주어진다. 질문마다 독립적으로 처리하고, 결과에 같은 번호를 붙인다."),
    ("human", "{numbered_questions}"),
])


def _clean(question: str, expanded: list[str], max_queries: int) -> list[str]:
    """원질문을 첫 번째로, 공백·중복 제거, 최대 1 + max_queries개"""
    queries = [question]
    for q in expanded:
        q = q.strip()
        if q and q not in queries:
            queries.append(q)
    return queries[: 1 + max_queries]


def _default_llm(**kwargs):
    from common.ai_model import get_llm_model

    return get_llm_model(**kwargs)


def _default_dense_fn() -> DenseFn:
    from retrieval.dense import dense_retrieve  # B 담당 모듈

    return dense_retrieve


def expand_queries(
    question: str,
    llm=None,
    max_queries: int = MAX_EXPANDED_QUERIES,
    before_call: Callable[[], None] | None = None,
    errors: list[str] | None = None,
) -> list[str]:
    """원질문을 첫 번째로 포함한 질의 목록을 반환 (최대 1 + max_queries개)

    LLM 호출이 실패하면 원질문만 반환한다. errors 리스트를 넘기면 실패 사유를 담는다
    (실패를 숨기면 Multi Query 결과가 baseline과 같아져도 알 수 없으므로).
    before_call: LLM 호출 직전에 실행 (평가의 호출 속도 제한용)
    """
    question = question.strip()
    if not question:
        raise ValueError("빈 질문입니다.")

    try:
        chain = EXPAND_PROMPT | (llm or _default_llm()).with_structured_output(
            ExpandedQueries, method="function_calling"
        )
        if before_call:
            before_call()
        expanded = chain.invoke({"question": question, "n": max_queries}).queries
    except Exception as e:
        logger.warning("질의 확장 실패, 원질문으로만 검색합니다.", exc_info=True)
        if errors is not None:
            errors.append(f"{type(e).__name__}: {e}"[:300])
        return [question]

    return _clean(question, expanded, max_queries)


def expand_queries_batch(
    questions: list[str],
    llm=None,
    max_queries: int = MAX_EXPANDED_QUERIES,
) -> dict[str, list[str]]:
    """여러 질문을 한 번의 LLM 요청으로 확장 → {질문: 질의 목록}

    응답에서 빠진 질문은 결과에 넣지 않는다 (호출한 쪽에서 개별 확장으로 보완).
    호출 실패는 예외를 그대로 올린다 (호출한 쪽에서 재시도·대체 처리).
    """
    questions = list(dict.fromkeys(q.strip() for q in questions if q.strip()))
    if not questions:
        return {}
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    if llm is None:
        # 기본 응답 길이(512토큰)로는 질문 7개 이상이면 응답이 잘려 빈 결과가 된다 (실측).
        # 질문당 확장 질의 3개 ≈ 150토큰으로 잡고 여유를 둔다.
        llm = _default_llm(max_tokens=min(4096, 300 + 250 * len(questions)))
    chain = EXPAND_BATCH_PROMPT | llm.with_structured_output(BatchExpandedQueries, method="function_calling")
    result = chain.invoke({"numbered_questions": numbered, "n": max_queries})
    if result is None:  # 구조화 출력 파싱 실패 (응답 잘림 등)
        raise ValueError("질의 확장 묶음 응답을 해석하지 못했습니다")
    expanded = {}
    for item in result.items:
        if 1 <= item.id <= len(questions):
            question = questions[item.id - 1]
            expanded[question] = _clean(question, item.queries, max_queries)
    return expanded


def rrf_merge(
    ranked_lists: list[list[RetrievedChunk]],
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[RetrievedChunk]:
    """여러 검색 결과를 chunk_id 기준으로 중복 제거하고 RRF 점수로 재정렬"""
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for position, chunk in enumerate(ranked, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + position)
            chunks.setdefault(chunk_id, chunk)

    merged_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        {
            **chunks[chunk_id],
            "retrieval_score": scores[chunk_id],  # RRF 점수 (Dense 유사도와 섞지 않음)
            "rerank_score": None,
            "rank": rank,
        }
        for rank, chunk_id in enumerate(merged_ids, start=1)
    ]


def multi_query_retrieve(
    question: str,
    candidate_k: int = CANDIDATE_K,
    per_query_k: int | None = None,
    dense_fn: DenseFn | None = None,
    llm=None,
    stats: dict | None = None,
    before_llm_call: Callable[[], None] | None = None,
    queries: list[str] | None = None,
    prefetch: Callable[[list[str]], None] | None = None,
    dense_many_fn: DenseManyFn | None = None,
) -> list[RetrievedChunk]:
    """Multi Query 검색

    per_query_k: 질의 하나당 Dense 후보 수 (기본값 candidate_k)
    stats: dict를 넘기면 사용한 질의와 호출 수를 기록한다 (평가용)
    queries: 이미 확장한 질의 목록 (평가에서 캐시된 확장 결과를 재사용할 때, 첫 번째는 원질문)
    prefetch: 질의들을 한 번의 요청으로 미리 임베딩하는 함수 (평가의 캐시용)
    dense_many_fn: 질의 목록을 받아 질의별 Dense 결과를 한 번에 돌려주는 함수.
        주면 dense_fn을 질의마다 부르지 않는다 (/ask에서 임베딩 4회 → 1회).
        예: B의 embed_queries(texts)로 벡터를 한 번에 받고, 벡터마다 Qdrant 검색
    """
    per_query_k = per_query_k or candidate_k

    errors: list[str] = []
    llm_calls = 0
    if queries is None:
        queries = expand_queries(question, llm=llm, before_call=before_llm_call, errors=errors)
        llm_calls = 1
    if prefetch:
        prefetch(queries)
    if dense_many_fn is not None:
        ranked_lists = dense_many_fn(queries, per_query_k)
        if len(ranked_lists) != len(queries):
            raise ValueError(f"dense_many_fn이 질의 {len(queries)}개에 결과 {len(ranked_lists)}개를 돌려줬습니다")
        dense_calls = 1
    else:
        dense_fn = dense_fn or _default_dense_fn()
        ranked_lists = [dense_fn(q, per_query_k) for q in queries]
        dense_calls = len(queries)
    results = rrf_merge(ranked_lists, top_k=candidate_k)

    if stats is not None:
        stats.update({
            "queries": queries,
            "llm_calls": llm_calls,
            "expand_prompt_version": EXPAND_PROMPT_VERSION,
            "expansion_failed": bool(errors),
            "expansion_error": errors[0] if errors else None,
            "dense_calls": dense_calls,
            "candidate_budget": per_query_k * len(queries),
            # 질의별 Dense 결과 (어떤 질의가 어떤 청크를 가져왔는지 확인용)
            "per_query": [
                {"query": q, "results": ranked}
                for q, ranked in zip(queries, ranked_lists)
            ],
        })
    return results
