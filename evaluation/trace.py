"""질문 하나를 모드별로 실행하고 과정 전체를 추적 (골든셋에 없는 임의 질문용)

질문 → (Multi Query면 확장 질의 → 질의별 Dense → RRF 병합) → (Rerank) → 최종 청크 → 답변·사용 청크

    from evaluation.trace import trace_question, trace_to_markdown
    trace = trace_question("고영향 인공지능이란?", modes=["baseline", "multi_query"], answer="temp")
    print(trace_to_markdown(trace))

터미널:
    uv run python -m evaluation.trace "고영향 인공지능이란?" --answer temp   # 기본: A의 컬렉션
    uv run python -m evaluation.trace "..." --gold 제2조   # 정답 조문을 알면 ✅/❌ 표시
    # 결과는 evaluation/results/trace_<시각>.md / .json 으로 저장
"""

import argparse
import json
import time
from datetime import datetime

from evaluation.chunk_source import resolve_collection
from evaluation.results_view import _short as results_short
from evaluation.results_view import mq_change, remember_labels
from evaluation.run_eval import (
    CANDIDATE_K,
    RERANK_MODES,
    NOT_CONNECTED_MESSAGE,
    RESULT_DIR,
    _answer_and_judge,
    _chunk_labels,
    _summarize_stats,
    get_dense_fn,
    get_generate_fn,
    is_relevant,
    rerank_available,
    run_mode,
)

DEFAULT_MODES = ["baseline", "rerank", "multi_query", "combined"]


def _short(chunk_id: str) -> str:
    return results_short(chunk_id)


def trace_question(
    question: str,
    modes: list[str] | None = None,
    final_k: int = 5,
    candidate_k: int = CANDIDATE_K,
    collection: str | None = None,
    answer: str | None = None,
    gold_articles: list[str] | None = None,
) -> dict:
    """질문 하나의 모드별 검색·답변 과정을 dict로 반환 (JSON 직렬화 가능)

    answer: None(검색만) / "temp"(C 임시 답변) / "shared"(B 답변 함수)
    gold_articles: 정답 조문을 알면 넣는다 (예: ["제2조"]). 없으면 정답 여부는 None
    """
    question = question.strip()
    if not question:
        raise ValueError("빈 질문입니다.")

    dense_fn = get_dense_fn(collection)
    generate_fn = get_generate_fn(answer)
    evidences = [{"article": a, "paragraph": None} for a in (gold_articles or [])]
    item = {"id": "trace", "question": question, "answerable": True, "evidences": evidences}

    def gold_flag(chunk):
        return is_relevant(chunk, evidences) if evidences else None

    by_mode = {}
    for mode in modes or DEFAULT_MODES:
        if mode in RERANK_MODES and not rerank_available():
            by_mode[mode] = {"status": "not_connected", "message": NOT_CONNECTED_MESSAGE}
            continue

        stats: dict = {}
        start = time.perf_counter()
        results = run_mode(mode, question, candidate_k, final_k, stats, dense_fn)
        retrieval_sec = round(time.perf_counter() - start, 3)

        retrieved = [
            {
                "rank": r["rank"],
                "chunk_id": r["chunk_id"],
                "article": r["article"],
                "article_title": r.get("article_title"),
                "score": round(r["retrieval_score"], 4),
                "rerank_score": r.get("rerank_score"),
                "is_gold": gold_flag(r),
                "content": r["content"],
            }
            for r in results
        ]
        labels = _chunk_labels(results, stats)
        remember_labels(labels)
        summary = _summarize_stats(stats, item)
        entry = {
            "status": "ok",
            "chunk_labels": labels,
            "retrieval_sec": retrieval_sec,
            "retrieved": retrieved,
            "queries": summary.get("queries"),
            "per_query": summary.get("per_query"),
            "candidate_ids": summary.get("candidate_ids"),
            "llm_calls": summary.get("llm_calls", 0),
        }
        if entry["per_query"]:
            entry["mq_change"] = mq_change(entry["per_query"], retrieved)
        if generate_fn is not None:
            record = _answer_and_judge(item, results, generate_fn, judge=False)
            for c in record["used_chunks"]:
                c["is_gold"] = gold_flag(c) if evidences else None
            entry["answer"] = record
        by_mode[mode] = entry

    return {
        "question": question,
        "gold_articles": gold_articles or [],
        "config": {"final_k": final_k, "candidate_k": candidate_k, "collection": resolve_collection(collection),
                   "answer_generator": answer or "none"},
        "modes": by_mode,
    }


def _mark(flag) -> str:
    return {True: " ✅", False: " ❌"}.get(flag, "")


def trace_to_markdown(trace: dict) -> str:
    for m in trace["modes"].values():  # 저장된 trace JSON을 다시 그릴 때도 조 번호로 표시
        remember_labels(m.get("chunk_labels", {}))
    lines = [f"# 질문 추적: {trace['question']}", ""]
    lines.append(f"- 설정: {json.dumps(trace['config'], ensure_ascii=False)}")
    if trace["gold_articles"]:
        lines.append(f"- 정답 조문: {', '.join(trace['gold_articles'])}")
    lines.append("")

    for mode, m in trace["modes"].items():
        lines.append(f"## {mode}")
        lines.append("")
        if m["status"] != "ok":
            lines += [f"⏸ {m['message']}", ""]
            continue

        if m.get("per_query"):
            lines.append("**1) 확장 질의와 질의별 Dense 결과**")
            lines.append("")
            for i, pq in enumerate(m["per_query"]):
                label = "원질문" if i == 0 else f"확장{i}"
                lines.append(f"- {label}: {pq['query']}")
                lines.append(f"  - 상위 5: {', '.join(_short(c) for c in pq['chunk_ids'][:5])}")
            lines.append("")
            change = m["mq_change"]
            lines.append("**2) 원질문 결과 대비 변화**")
            lines.append("")
            lines.append(f"- 원질문만 썼을 때: {', '.join(_short(c) for c in change['original'])}")
            lines.append(f"- 최종: {', '.join(_short(c) for c in change['final'])}")
            lines.append(f"- 새로 들어온 청크: {', '.join(_short(c) for c in change['added']) or '없음'}"
                         f" / 밀려난 청크: {', '.join(_short(c) for c in change['dropped']) or '없음'}")
            lines.append("")

        lines.append(f"**최종 검색 결과** ({m['retrieval_sec']}초, LLM 호출 {m['llm_calls']}회)")
        lines.append("")
        lines.append("| 순위 | 청크 | 검색 점수 | Rerank 점수 | 내용 |")
        lines.append("|---|---|---|---|---|")
        for r in m["retrieved"]:
            content = r["content"][:80].replace("\n", " ").replace("|", "/")
            rerank = "-" if r["rerank_score"] is None else round(r["rerank_score"], 4)
            lines.append(f"| {r['rank']} | {_short(r['chunk_id'])}{_mark(r['is_gold'])} | {r['score']} | {rerank} | {content} |")
        lines.append("")

        ans = m.get("answer")
        if ans:
            refuse = " (답변 불가로 응답)" if not ans["is_answerable"] else ""
            lines.append(f"**답변**{refuse} ({ans['generation_sec']}초)")
            lines.append("")
            lines.append("> " + ans["answer"].replace("\n", "\n> "))
            lines.append("")
            used = ", ".join(f"{_short(c['chunk_id'])}(검색 {c['rank']}위){_mark(c['is_gold'])}"
                             for c in ans["used_chunks"]) or "없음"
            lines.append(f"답변이 실제 사용한 청크: {used}")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--answer", choices=["temp", "shared"], default=None)
    parser.add_argument("--gold", nargs="*", default=None, help="정답 조문 (예: 제2조 제34조)")
    args = parser.parse_args()

    trace = trace_question(args.question, args.modes, args.k, collection=args.collection,
                           answer=args.answer, gold_articles=args.gold)
    markdown = trace_to_markdown(trace)
    print(markdown)

    RESULT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULT_DIR / f"trace_{stamp}.md").write_text(markdown, encoding="utf-8")
    (RESULT_DIR / f"trace_{stamp}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> evaluation/results/trace_{stamp}.md")


if __name__ == "__main__":
    main()
