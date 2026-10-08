"""골든셋으로 검색 모드(baseline/rerank)를 평가해 결과 JSON을 저장한다.

실행: uv run python -m scripts.eval_retrieval --mode rerank
결과: evaluation/results/sample/{mode}_{날짜-시각}.json (웹 /eval 페이지에서 확인)
형식은 evaluation/EVAL_PROTOCOL.md 8절을 따르고, 중간 과정 확인용 필드(retrieved, candidates)를 더한다.
"""
import argparse
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import openai

from common.config import CANDIDATE_K, COLLECTION_NAME, EMBEDDING_MODEL, FINAL_K, MODEL
from common.qdrant import get_qdrant_client
from retrieval.dense import dense_retrieve
EVAL_MODES = ("baseline", "rerank")  # 이 스크립트는 두 모드만 평가한다. multi_query 계열은 evaluation.run_eval 사용
from retrieval.rerank import rerank

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "evaluation" / "golden_set.json"
RESULT_DIR = ROOT / "evaluation" / "results" / "sample"


def collection_articles(collection: str) -> set[str]:
    """컬렉션에 실제로 들어 있는 조문 표기 집합. 정답이 없는 문항은 평가에서 제외하기 위해 쓴다."""
    client = get_qdrant_client()
    articles, offset = set(), None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True)
        articles |= {p.payload.get("article") for p in points}
        if offset is None:
            return articles


def unique_articles(chunks: list[dict]) -> list[str]:
    """순서를 유지한 채 조 단위로 중복을 제거한다 (프로토콜 4절)."""
    return list(dict.fromkeys(c.get("article") for c in chunks))


def score_question(retrieved: list[str], candidates: list[str], gold: list[str]) -> dict:
    ranks = [retrieved.index(g) + 1 for g in gold if g in retrieved]
    first = min(ranks) if ranks else None
    return {
        "hit@5": bool(ranks),
        "recall@5": len(ranks) / len(gold),
        "mrr": 1 / first if first else 0.0,
        "recall@candidate_k": sum(g in candidates for g in gold) / len(gold),
        "rank": first,
    }


def mean_metrics(rows: list[dict]) -> dict:
    keys = ("hit@5", "recall@5", "mrr", "recall@candidate_k")
    return {k: round(sum(float(r[k]) for r in rows) / len(rows), 4) for k in keys} if rows else {}


def chunk_view(c: dict, gold: list[str], with_content: bool) -> dict:
    view = {
        "rank": c.get("rank"),
        "article": c.get("article"),
        "title": c.get("article_title") or c.get("title"),
        "point_id": c.get("point_id"),
        "retrieval_score": c.get("retrieval_score"),
        "rerank_score": c.get("rerank_score"),
        "is_gold": c.get("article") in gold,
    }
    if with_content:
        view["content"] = c.get("content")
    return view


def git_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


class RerankRateLimited(Exception):
    """rerank가 호출 제한(429)으로 실패해 Dense 순서로 대체된 경우"""


RATE_LIMIT_WAIT_SEC = 30
RATE_LIMIT_MAX_RETRIES = 4


def search_once(question: str, mode: str, collection: str):
    """검색(+Rerank)을 한 번 수행하고 단계별 소요 시간을 함께 돌려준다."""
    t0 = time.perf_counter()
    candidates = dense_retrieve(question, candidate_k=CANDIDATE_K, collection=collection)
    t1 = time.perf_counter()
    final = rerank(question, candidates, final_k=FINAL_K) if mode == "rerank" else candidates[:FINAL_K]
    t2 = time.perf_counter()
    return candidates, final, t1 - t0, t2 - t1


def search_with_retry(question: str, mode: str, collection: str, stats: dict):
    """요청 한도(429)에 걸리면 같은 호출을 기다렸다가 다시 한다. 다른 방식으로 대체하지 않는다.

    시간은 성공한 시도만 재므로 대기 시간은 지연에 들어가지 않는다.
    """
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            candidates, final, r_sec, rr_sec = search_once(question, mode, collection)
            # rerank가 429 등으로 실패해 Dense 순서로 대체됐으면 rerank 결과로 기록하지 않고 기다렸다 다시 한다.
            if mode == "rerank" and final and all(c.get("rerank_score") is None for c in final):
                error = final[0].get("rerank_error", "")
                if "RateLimit" in error:
                    raise RerankRateLimited(error)
                raise RuntimeError(f"rerank 실패({error}): Dense 순서로 대체되어 평가에서 제외합니다")
            return candidates, final, r_sec, rr_sec
        except (openai.RateLimitError, RerankRateLimited):
            if attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            stats["rate_limit_retries"] += 1
            print(f"요청 한도 초과: {RATE_LIMIT_WAIT_SEC}초 뒤 같은 호출을 다시 시도합니다 ({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})")
            time.sleep(RATE_LIMIT_WAIT_SEC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=EVAL_MODES, default="rerank")
    parser.add_argument("--stage", choices=["base", "all"], default="all", help="base=기본 10문항, all=확장 포함")
    parser.add_argument("--split", choices=["tune", "final", "all"], default="all")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--max-rpm", type=int, default=20,
                        help="분당 모델 호출 상한. MonoRouter 한도(30)보다 낮게 잡아 다른 사람의 호출 여유를 남긴다")
    args = parser.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items = [
        q for q in golden["items"]
        if (args.stage == "all" or q.get("stage") == "base") and (args.split == "all" or q["split"] == args.split)
    ]
    available = collection_articles(args.collection)

    per_question, scored_rows = [], []
    latency = defaultdict(float)
    stats = {"rate_limit_retries": 0}
    calls_per_question = 2 if args.mode == "rerank" else 1  # 임베딩 1회 + (rerank면) LLM 1회
    min_gap = calls_per_question * 60 / args.max_rpm
    last_start = 0.0
    for q in items:
        gold = q["gold_articles"]
        entry = {"id": q["id"], "type": q["type"], "question": q["question"], "split": q["split"],
                 "answerable": q["answerable"], "gold_articles": gold}

        # 정답 조문이 컬렉션에 없으면 검색 성능이 아니라 데이터 부족이므로 지표에서 뺀다.
        missing = [g for g in gold if g not in available]
        if q["answerable"] and missing:
            per_question.append({**entry, "evaluated": False, "skip_reason": f"정답 조문이 컬렉션에 없음: {missing}"})
            continue

        # 분당 호출 수가 상한을 넘지 않도록 문항 사이 간격을 둔다. 이 대기는 지연 측정에 포함되지 않는다.
        time.sleep(max(0.0, last_start + min_gap - time.monotonic()))
        last_start = time.monotonic()
        candidates, final, retrieve_sec, rerank_sec = search_with_retry(q["question"], args.mode, args.collection, stats)
        latency["retrieve"] += retrieve_sec
        latency["rerank"] += rerank_sec

        entry |= {
            "retrieved_articles": unique_articles(final),
            "retrieved": [chunk_view(c, gold, with_content=True) for c in final],
            "candidates": [chunk_view(c, gold, with_content=False) for c in candidates],
        }
        if not q["answerable"]:
            per_question.append({**entry, "evaluated": False, "skip_reason": "답변 불가 문항(검색 지표에서 제외)"})
            continue

        scores = score_question(entry["retrieved_articles"], unique_articles(candidates), gold)
        per_question.append({**entry, "evaluated": True, **scores})
        scored_rows.append({**scores, "type": q["type"]})

    by_type = defaultdict(list)
    for r in scored_rows:
        by_type[r["type"]].append(r)

    run_calls = len([p for p in per_question if "retrieved" in p])
    result = {
        "experiment": args.mode,
        "config": {
            "collection": args.collection, "data_version": "sample" if "sample" in args.collection or len(available) <= 10 else "v1",
            "collection_articles": sorted(available),
            "embedding_model": EMBEDDING_MODEL, "chat_model": MODEL,
            "rerank": "LLM 기반" if args.mode == "rerank" else None,
            "candidate_k": CANDIDATE_K, "final_k": FINAL_K,
            "golden_set_version": golden["meta"]["version"], "stage": args.stage, "split": args.split,
            "max_rpm": args.max_rpm,
            "code_version": git_version(), "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "counts": {"total": len(items), "evaluated": len(scored_rows), "skipped": len(items) - len(scored_rows),
                   "rate_limit_retries": stats["rate_limit_retries"]},
        "metrics": {"overall": mean_metrics(scored_rows), "by_type": {t: mean_metrics(r) for t, r in by_type.items()}},
        "cost": {
            "llm_calls_per_query": 1 if args.mode == "rerank" else 0,
            "latency_sec": {
                "retrieve": round(latency["retrieve"] / max(run_calls, 1), 3),
                "rerank": round(latency["rerank"] / max(run_calls, 1), 3),
                "total": round((latency["retrieve"] + latency["rerank"]) / max(run_calls, 1), 3),
            },
        },
        "per_question": per_question,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"{args.mode}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {path.relative_to(ROOT)}")
    print(f"평가 {len(scored_rows)}/{len(items)}문항 · overall {result['metrics']['overall']}")


if __name__ == "__main__":
    main()
