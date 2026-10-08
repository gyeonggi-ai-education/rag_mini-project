"""동일 Golden Set으로 검색 모드별 지표를 계산한다 (C 담당)

사용법:
    # 개발 중: 검색만 (첫 실행 = 확장 1회 + 임베딩 1회, 재실행 0회)
    uv run python -m evaluation.run_eval --k 5
    # 최종: 답변은 baseline·multi_query만, 채점은 규칙 기반(API 0회) (B 답변 함수 전에는 --answer temp)
    uv run python -m evaluation.run_eval --k 5 --answer shared
    # LLM 채점까지: --judge llm (문항당 1회), 답변 모드 변경: --answer-modes baseline rerank multi_query
    # 빠른 확인: --limit 3, 튜닝/최종 확인 분리: --split tune|test, 캐시 끄기: --no-cache

API 호출 절감: 확장·임베딩은 prepare()에서 묶어서 한 번에, 모든 호출 결과는 evaluation/.cache 에 저장

모드: baseline, rerank, multi_query, combined
B의 dense_retrieve / rerank가 아직 없으면 해당 모드는 import 에러로 실패한다.
기본: 실제 골든셋 10문항(golden_set.jsonl) + A의 컬렉션(data_ingestion_law21311_v1).
더미 확인이 필요할 때만 --golden evaluation/dummy/golden_set.dummy.jsonl --collection tmp_c_law_articles
"""

import argparse
import hashlib
import json
import subprocess
import time
from functools import partial
from datetime import datetime
from pathlib import Path

from common.config import EMBEDDING_MODEL, MODEL
from evaluation.answer_judge import JUDGE_VERSION, RULE_JUDGE_VERSION, rule_judge
from retrieval.multi_query import EXPAND_PROMPT_VERSION
from evaluation.chunk_source import resolve_collection
from evaluation.metrics import hit_at_k, is_relevant, recall_at_k, reciprocal_rank
from evaluation.rate_limit import RATE_LIMIT_RETRIES, RATE_LIMIT_WAIT_SEC, call_limited, limited, limiter
from langchain_core.callbacks import get_usage_metadata_callback

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_GOLDEN = EVAL_DIR / "golden_set.jsonl"
RESULT_DIR = EVAL_DIR / "results"

CANDIDATE_K = 20
RERANK_INPUT_K = 20  # combined 모드에서 Rerank 입력 후보 수 제한


_CACHES: dict = {}


def setup_cache(enabled: bool = True) -> dict:
    """호출 절감용 캐시 설치 (evaluation/cache.py 참고)

    B의 retrieval/dense.py 파일은 그대로 두고, 실행 중에만 임베딩 객체를 캐시로 감싼다.
    호출 속도 제한은 실제 요청이 나갈 때(캐시 미적중)만 적용된다.
    """
    import retrieval.dense as dense
    from evaluation.cache import CachedEmbeddings, JsonCache

    for name in ("embeddings", "expansions", "answers", "judgements", "reranks"):
        _CACHES[name] = JsonCache(name, enabled)
    inner = getattr(dense._embedding, "inner", dense._embedding)
    dense._embedding = CachedEmbeddings(inner, _CACHES["embeddings"], EMBEDDING_MODEL, call=call_limited)
    return _CACHES


def save_caches():
    for cache in _CACHES.values():
        cache.save()


def cache_stats() -> dict:
    return {name: {"hits": c.hits, "misses": c.misses} for name, c in _CACHES.items() if c.enabled}


def _cache(name: str):
    """설치된 캐시, 없으면 비활성 캐시 (가짜 함수 테스트 등)"""
    from evaluation.cache import JsonCache

    return _CACHES.get(name) or JsonCache(name, enabled=False)


def _prefetch_fn():
    import sys

    dense = sys.modules.get("retrieval.dense")
    embedding = getattr(dense, "_embedding", None) if dense else None
    return getattr(embedding, "prefetch", None)


def get_dense_fn(collection: str | None = None):
    """B의 공통 dense_retrieve로 검색할 컬렉션을 고정한 함수

    collection 생략 시 QDRANT_COLLECTION 환경변수, 없으면 A의 컬렉션 (evaluation/chunk_source.py).
    B의 dense_retrieve 기본값에 기대지 않고 항상 명시해서, 평가 기록의 컬렉션과 실제 검색 대상이 같게 한다.
    """
    if not _CACHES:
        setup_cache(enabled=True)
    from evaluation.chunk_source import resolve_collection
    from retrieval.dense import dense_retrieve

    # 임베딩 캐시가 실제 요청 때만 호출 제한을 거므로 여기서는 감싸지 않는다
    return partial(dense_retrieve, collection=resolve_collection(collection))


MODES = ["baseline", "rerank", "multi_query", "combined"]
RERANK_MODES = {"rerank", "combined"}
NOT_CONNECTED_MESSAGE = "Rerank가 아직 연결되지 않았습니다 (retrieval/rerank.py 의 rerank 함수 필요)"


def rerank_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("retrieval.rerank") is not None


def not_connected_result(mode: str, final_k: int, candidate_k: int) -> dict:
    """Rerank 미연결 모드: 결과 화면에 자리만 표시되도록 상태만 기록"""
    return {
        "summary": {"mode": mode, "status": "not_connected", "message": NOT_CONNECTED_MESSAGE,
                    "final_k": final_k, "candidate_k": candidate_k},
        "rows": [],
    }
RERANK_REQUIRED_KEYS = {"chunk_id", "article", "content", "retrieval_score", "rerank_score", "rank"}


def _strict_rerank(question: str, candidates: list[dict], final_k: int) -> list[dict]:
    """B의 rerank는 실패하면 Dense 순서로 대체하고 rerank_score를 모두 None으로 남긴다.
    평가에서는 이를 rerank 결과로 기록·캐시하지 않고 실패로 올린다(호출 제한이면 call_limited가 기다렸다 재시도)."""
    from retrieval.rerank import rerank

    results = rerank(question, candidates, final_k)
    if results and all(r.get("rerank_score") is None for r in results):
        error = results[0].get("rerank_error", "unknown")
        raise RuntimeError(("429 " if "RateLimit" in error else "") + f"rerank 실패로 Dense 순서 대체됨: {error}")
    return results


def _rerank(question: str, candidates: list[dict], final_k: int, stats: dict) -> list[dict]:
    """B의 rerank 호출 + Rerank 전 후보·지연 기록 (후보 누락과 순위 문제를 구별하기 위함)"""
    from retrieval.rerank import rerank

    stats["candidates"] = candidates
    # 같은 질문 + 같은 후보면 결과 재사용 (재실행 시 Rerank 호출 0회)
    cache = _cache("reranks")
    key = cache.key("rerank", getattr(rerank, "__module__", ""), question, [c["chunk_id"] for c in candidates], final_k)
    start = time.perf_counter()
    results = cache.get(key)
    stats["rerank_cached"] = results is not None
    if results is None:
        # LLM 기반 Rerank가 내부에서 여러 번 호출하면 실제 요청 수는 더 많을 수 있음
        results = call_limited(_strict_rerank, question, candidates, final_k)
        cache.set(key, results)
    stats["rerank_sec"] = round(time.perf_counter() - start, 3)

    for r in results:
        missing = RERANK_REQUIRED_KEYS - r.keys()
        if missing:
            raise ValueError(f"rerank 결과에 필드 누락: {missing} (schemas/chunk.py RetrievedChunk 형식 확인)")
    return results


def run_mode(mode: str, question: str, candidate_k: int, final_k: int, stats: dict, dense_fn) -> list[dict]:
    if mode == "baseline":
        stats["dense_calls"] = 1
        return dense_fn(question, final_k)

    if mode == "rerank":
        stats["dense_calls"] = 1
        candidates = dense_fn(question, candidate_k)
        return _rerank(question, candidates, final_k, stats)

    if mode == "multi_query":
        return _multi_query(question, candidate_k, dense_fn, stats)[:final_k]

    if mode == "combined":
        candidates = _multi_query(question, candidate_k, dense_fn, stats)[:RERANK_INPUT_K]
        return _rerank(question, candidates, final_k, stats)

    raise ValueError(f"알 수 없는 모드: {mode}")


def _expansion_key(question: str) -> str:
    from evaluation.cache import JsonCache

    return JsonCache.key("expand", EXPAND_PROMPT_VERSION, MODEL, question)


MQ_MODES = {"multi_query", "combined"}
EXPAND_BATCH_SIZE = 10


def prepare(golden: list[dict], modes: list[str]) -> dict:
    """평가 시작 전 API 호출을 묶어서 처리

    1. 질의 확장: 캐시에 없는 질문들을 EXPAND_BATCH_SIZE개씩 한 번의 요청으로 확장
    2. 임베딩: 모든 질문 + 확장 질의를 한 번의 요청으로 임베딩
    이후 모드별 평가는 캐시만 읽으므로 확장·임베딩 요청이 생기지 않는다.
    """
    from retrieval.multi_query import expand_queries_batch

    requests_before = limiter.count
    questions = list(dict.fromkeys(g["question"] for g in golden))
    texts = list(questions)
    info = {"batch_expanded": 0, "batch_failed": []}

    if MQ_MODES & set(modes):
        cache = _cache("expansions")
        missing = [q for q in questions if cache.get(_expansion_key(q)) is None]
        pending = [missing[i:i + EXPAND_BATCH_SIZE] for i in range(0, len(missing), EXPAND_BATCH_SIZE)]
        while pending:
            batch = pending.pop(0)
            try:
                expanded = call_limited(expand_queries_batch, batch)
            except Exception as e:
                info["batch_failed"].append(f"{len(batch)}문항: {type(e).__name__}: {e}"[:200])
                expanded = {}
                if len(batch) > 1:  # 반으로 나눠 다시 시도, 1문항도 실패하면 평가 중 개별 확장으로 보완
                    half = len(batch) // 2
                    pending[:0] = [batch[:half], batch[half:]]
            for question, queries in expanded.items():
                cache.set(_expansion_key(question), queries)
            info["batch_expanded"] += len(expanded)
        for q in questions:
            texts += cache.get(_expansion_key(q)) or []

    prefetch = _prefetch_fn()
    if prefetch:
        prefetch(texts)
    save_caches()
    info["requests"] = limiter.count - requests_before
    return info


def _expanded_queries(question: str, stats: dict) -> list[str]:
    """질의 확장 (캐시): multi_query와 combined가 같은 확장 질의를 쓰고, 재실행 때 다시 호출하지 않는다

    보통 prepare()에서 묶어서 확장해 두므로 여기서는 캐시를 읽기만 한다.
    """
    from retrieval.multi_query import expand_queries

    cache = _cache("expansions")
    key = _expansion_key(question)
    cached = cache.get(key)
    if cached is not None:
        stats["expansion_cached"] = True
        return cached

    # expand_queries는 실패 시 원질문으로 대체하므로, 429면 여기서 기다렸다가 다시 시도한다
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        errors: list[str] = []
        queries = expand_queries(question, before_call=limiter.acquire, errors=errors)
        stats["expansion_llm_calls"] = stats.get("expansion_llm_calls", 0) + 1
        if not errors or "RateLimitError" not in errors[0] or attempt == RATE_LIMIT_RETRIES:
            break
        limiter.rate_limited += 1
        time.sleep(RATE_LIMIT_WAIT_SEC)
        limiter.waited_sec += RATE_LIMIT_WAIT_SEC
    if errors:
        stats["expansion_failed"] = True
        stats["expansion_error"] = errors[0]
    else:
        cache.set(key, queries)  # 실패한 결과(원질문만)는 저장하지 않음
    return queries


def _multi_query(question: str, candidate_k: int, dense_fn, stats: dict) -> list[dict]:
    from retrieval.multi_query import multi_query_retrieve

    queries = _expanded_queries(question, stats)
    expansion = {k: stats.pop(k) for k in ("expansion_cached", "expansion_llm_calls", "expansion_failed",
                                           "expansion_error") if k in stats}
    results = multi_query_retrieve(question, candidate_k, dense_fn=dense_fn, stats=stats,
                                   queries=queries, prefetch=_prefetch_fn())
    stats.update(expansion)
    stats["llm_calls"] = expansion.get("expansion_llm_calls", 0)
    stats["expansion_failed"] = expansion.get("expansion_failed", False)
    stats["expansion_error"] = expansion.get("expansion_error")
    return results


def _chunk_labels(results: list[dict], stats: dict) -> dict[str, str]:
    """이 문항에서 본 모든 청크의 표시 이름 (최종 결과, Rerank 전 후보, 질의별 결과)"""
    from evaluation.results_view import chunk_label

    chunks = list(results) + list(stats.get("candidates") or [])
    for pq in stats.get("per_query") or []:
        chunks += pq["results"]
    return {c["chunk_id"]: chunk_label(c) for c in chunks}


def _first_gold_rank(results: list[dict], evidences: list[dict]) -> int | None:
    return next((i for i, r in enumerate(results, start=1) if is_relevant(r, evidences)), None)


def _summarize_stats(stats: dict, item: dict) -> dict:
    """원본 청크가 담긴 stats를 결과 파일용으로 줄이고 정답 위치를 계산한다"""
    evidences = item["evidences"] if item["answerable"] else []
    stats = dict(stats)

    candidates = stats.pop("candidates", None)
    if candidates is not None:
        stats["candidate_ids"] = [c["chunk_id"] for c in candidates]
        if evidences:
            stats["candidate_gold_rank"] = _first_gold_rank(candidates, evidences)
            stats["candidate_recall"] = recall_at_k(candidates, evidences, len(candidates))

    if "per_query" in stats:
        stats["per_query"] = [
            {
                "query": pq["query"],
                "chunk_ids": [c["chunk_id"] for c in pq["results"]],
                "gold_rank": _first_gold_rank(pq["results"], evidences) if evidences else None,
            }
            for pq in stats["per_query"]
        ]
    return stats


def get_generate_fn(source: str | None):
    """temp: C 임시 답변 생성 / shared: B의 답변 생성 함수 / None: 답변 생성 안 함"""
    if source is None:
        return None
    if source == "temp":
        from evaluation.temp_generate import generate_answer
    elif source == "shared":
        from generation.answer import generate_answer  # B 담당 (계획서 디렉터리 기준, 이름은 B와 확정)
    else:
        raise ValueError(f"알 수 없는 답변 생성기: {source}")
    return limited(generate_answer)


def _answer_and_judge(item: dict, results: list[dict], generate_fn, judge: str | None) -> dict:
    """검색 결과로 답변을 만들고, 답변이 실제로 사용한 청크와 채점 결과를 기록한다"""
    # 같은 질문 + 같은 청크 조합이면 답변 재사용 (예: baseline과 rerank의 최종 청크가 같을 때)
    answer_cache = _cache("answers")
    generator = getattr(generate_fn, "__wrapped__", generate_fn).__module__
    answer_key = answer_cache.key("answer", generator, MODEL, item["question"], [r["chunk_id"] for r in results])
    start = time.perf_counter()
    answer = answer_cache.get(answer_key)
    answer_cached = answer is not None
    if not answer_cached:
        answer = generate_fn(item["question"], results)
        answer_cache.set(answer_key, answer)
    generation_sec = round(time.perf_counter() - start, 3)

    by_id = {r["chunk_id"]: r for r in results}
    used_chunks = [by_id[c] for c in answer.get("used_chunk_ids", []) if c in by_id]
    evidences = item["evidences"] if item["answerable"] else []
    record = {
        "answer": answer["answer"],
        "is_answerable": answer["is_answerable"],
        "used_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "article": c["article"],
                "rank": c["rank"],
                "is_gold": is_relevant(c, evidences) if evidences else False,
                "content": c["content"][:80],
            }
            for c in used_chunks
        ],
        "used_gold": any(is_relevant(c, evidences) for c in used_chunks) if evidences else None,
        "generation_sec": generation_sec,
        "answer_cached": answer_cached,
        "answer_llm_calls": 0 if answer_cached else 1,
    }
    if judge in ("rule", "llm"):
        # 규칙 기반 채점은 API 호출 없음. LLM 채점은 모든 모드가 끝난 뒤 문항 단위로 묶어서 (llm_judge_pass)
        record["judgement"] = rule_judge(item, answer, used_chunks)
        record["_used_full"] = used_chunks  # LLM 채점용 근거 원문 (저장 전에 제거)
    return record


def _token_totals(usage_metadata: dict) -> dict:
    """모델별 사용량을 합산 (비용 계산은 모델 단가를 곱해서 보고서에서)"""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in usage_metadata.values():
        for key in totals:
            totals[key] += usage.get(key, 0)
    return totals


def _answer_summary(rows: list[dict]) -> dict:
    """답변 지표: 답변 가능 문항과 답변 불가 문항을 따로 집계"""
    answered = [r for r in rows if "answer" in r]
    if not answered:
        return {}

    def avg(values):
        values = list(values)
        return round(sum(values) / len(values), 3) if values else None

    ok = [r for r in answered if r["answerable"]]
    ng = [r for r in answered if not r["answerable"]]
    rule = [r["answer"]["judgement"] for r in ok if "key_point_coverage" in r["answer"].get("judgement", {})]
    judged = [r for r in ok if "correctness" in (r["answer"].get("llm_judgement") or {})]
    summary = {
        "avg_generation_sec": avg(r["answer"]["generation_sec"] for r in answered),
        "avg_total_sec": avg(r["latency_sec"] + r["answer"]["generation_sec"] for r in answered),
        # 답변이 정답 근거를 실제로 사용했는지 / 답변 가능한데 거절했는지
        "used_gold_rate": avg(float(r["answer"]["used_gold"]) for r in ok),
        "false_refusal_rate": avg(float(not r["answer"]["is_answerable"]) for r in ok),
    }
    if ng:
        summary["unanswerable_refusal_rate"] = avg(float(not r["answer"]["is_answerable"]) for r in ng)
    if rule:
        summary.update({
            "avg_key_point_coverage": avg(x["key_point_coverage"] for x in rule if x["key_point_coverage"] is not None),
            "avg_grounded_ratio": avg(x["grounded_ratio"] for x in rule if x["grounded_ratio"] is not None),
            "ungrounded_answer_rate": avg(float(bool(x["ungrounded_sentences"])) for x in rule),
        })
    if judged:
        j = [r["answer"]["llm_judgement"] for r in judged]
        summary.update({
            "avg_correctness": avg(x["correctness"] for x in j),
            "avg_relevance": avg(x["relevance"] for x in j),
            "avg_faithfulness": avg(x["faithfulness"] for x in j),
            "unsupported_claim_rate": avg(float(bool(x["unsupported_claims"])) for x in j),
        })
    return summary


def llm_judge_pass(results: dict[str, dict], golden: list[dict]) -> int:
    """모든 모드가 끝난 뒤 문항마다 한 번의 호출로 여러 모드의 답변을 함께 LLM 채점

    같은 답변(같은 문장 + 같은 근거)은 한 번만 채점한다. 반환: 실제 LLM 요청 수
    """
    from evaluation.answer_judge import JUDGE_VERSION as LLM_JUDGE_VERSION
    from evaluation.answer_judge import llm_judge_batch

    cache = _cache("judgements")
    requests = 0
    items = {g["id"]: g for g in golden}
    for qid, item in items.items():
        if not item["answerable"]:
            continue
        rows = [row for run in results.values() for row in run["rows"]
                if row["id"] == qid and "answer" in row and row["answer"]["is_answerable"]]
        pending: dict[str, tuple[dict, list[dict]]] = {}
        for row in rows:
            ans = row["answer"]
            key = cache.key("llm-judge", LLM_JUDGE_VERSION, MODEL, item["question"], item.get("key_points", []),
                            ans["answer"], [c["chunk_id"] for c in ans["used_chunks"]])
            ans["_judge_key"] = key
            if cache.get(key) is None and key not in pending:
                pending[key] = ({"answer": ans["answer"]}, ans.get("_used_full", []))
        if pending:
            keys = list(pending)
            judged = call_limited(llm_judge_batch, item, [pending[k] for k in keys])
            requests += 1
            for key, judgement in zip(keys, judged):
                if judgement is not None:
                    cache.set(key, judgement)
        for row in rows:
            row["answer"]["llm_judgement"] = cache.get(row["answer"].pop("_judge_key"))
    return requests


def _strip_internal(results: dict[str, dict]):
    """결과 파일에 저장하지 않을 내부 필드 제거"""
    for run in results.values():
        for row in run["rows"]:
            if "answer" in row:
                row["answer"].pop("_used_full", None)
                row["answer"].pop("_judge_key", None)


def load_golden(path: Path) -> list[dict]:
    """golden_set.jsonl 로딩 (정답은 조 번호 기준, golden_set.md 참고)

    answerable이 없으면 답변 가능 문항으로 본다.
    answer_chunk_ids가 있으면 조 번호 대신 청크 단위로 판정한다 (더미 골든셋).
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("answerable", True)
            if item.get("answer_chunk_ids"):
                item["evidences"] = [{"chunk_id": c} for c in item["answer_chunk_ids"]]
            else:
                item["evidences"] = [{"article": a, "paragraph": None} for a in item.get("answer_articles", [])]
            items.append(item)
    return items


def _average(rows: list[dict], final_k: int) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        f"hit@{final_k}": round(sum(r["hit"] for r in rows) / n, 4),
        f"recall@{final_k}": round(sum(r["recall"] for r in rows) / n, 4),
        "mrr": round(sum(r["rr"] for r in rows) / n, 4),
    }


def evaluate(mode: str, golden: list[dict], candidate_k: int, final_k: int, dense_fn,
             generate_fn=None, judge: str | None = "rule") -> dict:
    rows = []
    requests_before = limiter.count
    for item in golden:
        stats: dict = {}
        with get_usage_metadata_callback() as usage:  # LLM 토큰 사용량 (임베딩 토큰은 포함 안 됨)
            start = time.perf_counter()
            results = run_mode(mode, item["question"], candidate_k, final_k, stats, dense_fn)
            latency = time.perf_counter() - start
            answer_record = _answer_and_judge(item, results, generate_fn, judge) if generate_fn else None

        labels = _chunk_labels(results, stats)
        row = {
            "id": item["id"],
            "type": item.get("type"),
            "answerable": item["answerable"],
            "latency_sec": round(latency, 3),
            "question": item["question"],
            "gold": item.get("answer_chunk_ids") or item.get("answer_articles", []),
            "retrieved": [
                {
                    "rank": r["rank"],
                    "chunk_id": r["chunk_id"],
                    "article": r["article"],
                    "score": round(r["retrieval_score"], 4),
                    "rerank_score": r.get("rerank_score"),
                    "is_gold": is_relevant(r, item["evidences"]) if item["answerable"] else False,
                    "content": r["content"][:80],
                }
                for r in results
            ],
            **_summarize_stats(stats, item),
        }
        row["chunk_labels"] = labels
        if answer_record is not None:
            row["answer"] = answer_record
        row["tokens"] = _token_totals(usage.usage_metadata)
        if item["answerable"]:
            evidences = item["evidences"]
            row["hit"] = hit_at_k(results, evidences, final_k)
            row["recall"] = recall_at_k(results, evidences, final_k)
            row["rr"] = reciprocal_rank(results, evidences, final_k)
        rows.append(row)

    # 답변 불가 문항은 Retrieval 지표 평균에서 제외
    scored = [r for r in rows if r["answerable"]]
    types = sorted({r["type"] for r in scored if r["type"]})
    summary = {
        "mode": mode,
        "final_k": final_k,
        "candidate_k": candidate_k,
        "num_unanswerable": len(rows) - len(scored),
        **_average(scored, final_k),
        "avg_latency_sec": round(sum(r["latency_sec"] for r in rows) / len(rows), 3),
        "avg_llm_calls": round(sum(r.get("llm_calls", 0) for r in rows) / len(rows), 2),
        # 질의 확장 실패 수: 0이 아니면 Multi Query 결과가 baseline과 같아졌을 수 있음
        "expansion_failures": sum(1 for r in rows if r.get("expansion_failed")),
        "total_tokens": sum(r["tokens"]["total_tokens"] for r in rows),
        "avg_tokens": round(sum(r["tokens"]["total_tokens"] for r in rows) / len(rows), 1),
        # 유형별(기준/MQ/RR/둘다) 지표: 어떤 기법이 어떤 질문에 효과가 있었는지 비교용
        "by_type": {t: _average([r for r in scored if r["type"] == t], final_k) for t in types},
    }
    candidate_rows = [r for r in scored if "candidate_recall" in r]
    if candidate_rows:
        # Rerank 전 후보에 정답이 있었는지: 낮으면 후보 누락, 높은데 최종 지표가 낮으면 순위 문제
        summary[f"candidate_recall@{candidate_k}"] = round(
            sum(r["candidate_recall"] for r in candidate_rows) / len(candidate_rows), 4
        )
    answer_summary = _answer_summary(rows)
    if answer_summary:
        summary["answer"] = answer_summary
    summary["rate_limit_wait_sec"] = round(limiter.waited_sec, 1)  # 누적값 (지연 해석용)
    # 이 모드에서 실제로 MonoRouter에 보낸 요청 수 (캐시 적중 제외, Rerank 내부 추가 호출은 B 구현에 따라 더 많을 수 있음)
    summary["monorouter_requests"] = limiter.count - requests_before
    summary["rate_limited_total"] = limiter.rate_limited  # 누적 429 횟수 (같은 키를 쓰는 다른 사람 영향 포함)
    summary["status"] = "ok"
    return {"summary": summary, "rows": rows}


def _code_version() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=EVAL_DIR.parent, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=EVAL_DIR.parent).stdout.strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def detect_data_version(collection: str) -> str:
    """컬렉션 payload의 문서·청킹 버전으로 데이터 버전 표기 (Qdrant만 조회, API 호출 없음)"""
    try:
        from common.qdrant import get_qdrant_client

        points, _ = get_qdrant_client().scroll(collection_name=collection, limit=1, with_payload=True)
        payload = points[0].payload if points else {}
    except Exception:
        return "unknown"
    parts = [payload.get(k) for k in ("document_id", "chunking_version", "schema_version") if payload.get(k)]
    return ":".join(parts) or "unknown"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


DEFAULT_ANSWER_MODES = ["baseline", "multi_query"]  # 개선 전(baseline)과 C의 최종 모드만 답변 (호출 절감)


def run_all(golden: list[dict], modes: list[str], dense_fn, candidate_k: int, final_k: int,
            generate_fn=None, answer_modes: list[str] | None = None, judge: str | None = "rule") -> dict:
    """prepare(확장·임베딩 묶음) → 모드별 평가 → (선택) 문항 단위 LLM 채점

    answer_modes에 있는 모드만 답변을 생성한다. 반환: {"results": {모드: 결과}, "prepare": 정보}
    """
    prepare_info = prepare(golden, modes)
    answer_modes = modes if answer_modes is None else answer_modes
    results = {}
    for mode in modes:
        if mode in RERANK_MODES and not rerank_available():
            results[mode] = not_connected_result(mode, final_k, candidate_k)
            continue
        mode_generate = generate_fn if mode in answer_modes else None
        results[mode] = evaluate(mode, golden, candidate_k, final_k, dense_fn, mode_generate, judge)
        save_caches()

    if judge == "llm" and generate_fn is not None:
        requests_before = limiter.count
        prepare_info["llm_judge_requests"] = llm_judge_pass(results, golden)
        prepare_info["llm_judge_monorouter_requests"] = limiter.count - requests_before
        for run in results.values():  # LLM 채점이 붙었으니 답변 지표 다시 계산
            if run["rows"] and _answer_summary(run["rows"]):
                run["summary"]["answer"] = _answer_summary(run["rows"])
        save_caches()
    _strip_internal(results)
    return {"results": results, "prepare": prepare_info}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--modes", nargs="+", default=MODES,
                        help=f"{' '.join(MODES)} 또는 all")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--data-version", default=None,
                        help="데이터 버전 (생략 시 컬렉션 payload의 document_id:chunking_version:schema_version)")
    parser.add_argument("--collection", default=None,
                        help="검색할 Qdrant 컬렉션 (생략 시 QDRANT_COLLECTION, 없으면 A의 data_ingestion_law21311_v1. "
                             "더미 평가는 tmp_c_law_articles)")
    parser.add_argument("--answer", choices=["temp", "shared"], default=None,
                        help="답변 생성: temp(C 임시) / shared(B 함수). 생략 시 검색만 평가")
    parser.add_argument("--answer-modes", nargs="+", default=DEFAULT_ANSWER_MODES,
                        help="답변을 생성할 모드 (기본: baseline multi_query, 전부는 all)")
    parser.add_argument("--judge", choices=["none", "rule", "llm"], default="rule",
                        help="답변 채점: rule(기본, API 호출 없음) / llm(규칙 + 문항당 1회 LLM 채점) / none")
    parser.add_argument("--split", choices=["tune", "test"], default=None, help="골든셋 split 필터")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N문항만 (빠른 확인용)")
    parser.add_argument("--no-cache", action="store_true",
                        help="캐시를 쓰지 않고 모두 새로 호출 (LLM 결과 변동을 확인할 때)")
    args = parser.parse_args()

    modes = MODES if args.modes == ["all"] else args.modes
    answer_modes = MODES if args.answer_modes == ["all"] else args.answer_modes
    golden = load_golden(args.golden)
    if args.split:
        golden = [g for g in golden if g.get("split") == args.split]
    if args.limit:
        golden = golden[: args.limit]
    setup_cache(enabled=not args.no_cache)
    generate_fn = get_generate_fn(args.answer)
    dense_fn = get_dense_fn(args.collection)
    judge = None if args.judge == "none" else args.judge
    data_version = args.data_version or detect_data_version(resolve_collection(args.collection))
    RESULT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run = run_all(golden, modes, dense_fn, args.candidate_k, args.k, generate_fn, answer_modes, judge)
    print(f"준비 단계: {json.dumps(run['prepare'], ensure_ascii=False)}")
    for mode, result in run["results"].items():
        result["config"] = {
            "golden_set": args.golden.name,
            "golden_set_hash": _file_hash(args.golden),
            "split": args.split or "all",
            "num_questions": len(golden),
            "data_version": data_version,
            "collection": resolve_collection(args.collection),
            "llm_model": MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "answer_generator": (args.answer if mode in answer_modes else None) or "none",
            "answer_modes": answer_modes if args.answer else [],
            "judge": args.judge if args.answer else "none",
            "judge_version": {"rule": RULE_JUDGE_VERSION, "llm": f"{RULE_JUDGE_VERSION}+{JUDGE_VERSION}"}.get(args.judge)
                             if args.answer else None,
            "expand_prompt_version": EXPAND_PROMPT_VERSION,
            "code_version": _code_version(),
            "cache": "off" if args.no_cache else "on",
            "prepare": run["prepare"],
            **{k: v for k, v in result["summary"].items() if k in ("mode", "final_k", "candidate_k")},
        }
        out = RESULT_DIR / f"{stamp}_{mode}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result["summary"].items() if k != "by_type"}, ensure_ascii=False))
        print(f"  -> {out.relative_to(EVAL_DIR.parent)}")
    print(f"MonoRouter 요청 합계 {limiter.count}회 (429 {limiter.rate_limited}회), "
          f"캐시 {json.dumps(cache_stats(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
