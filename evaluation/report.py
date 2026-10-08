"""평가 결과를 질문별로 확인하는 마크다운 리포트 생성

- 모드별 최종 Top-K 청크와 정답 여부(✅/❌), 검색 점수·Rerank 점수
- Multi Query: 확장 질의, 질의별 Dense 결과에서 정답 청크가 몇 위였는지
- Rerank: Rerank 전 후보에서 정답이 몇 위였는지 (후보 누락 vs 순위 문제 구별)
- baseline 대비 모드별 개선/악화/동일 판정

실행:
    uv run python -m evaluation.report                 # 실제 골든셋 실행 중 가장 최근
    uv run python -m evaluation.report 20261008_120638 # 특정 실행
결과: evaluation/results/<stamp>_report.md
"""

import argparse
import json
import sys

from evaluation.results_view import (ANSWER_COLUMNS, RESULT_DIR, _short, build_results, describe_judgement,
                                     latest_stamp, load_runs, mq_change)
from evaluation.results_view import verdict as _verdict

PER_QUERY_SHOW = 5  # 질의별로 보여줄 Dense 상위 개수


def short(chunk_id: str) -> str:
    return _short(chunk_id)


def verdict(base_rr, mode_rr) -> str:
    return {"개선": "⬆️ 개선", "악화": "⬇️ 악화", "동일": "➖ 동일"}.get(_verdict(base_rr, mode_rr), "")


def build_report(stamp: str, runs: dict[str, dict]) -> str:
    chunk_tables = {q["id"]: q["chunks"] for q in build_results(stamp, runs)["questions"]}
    # Rerank 미연결 등 결과가 없는 모드는 자리만 표시
    pending = {m: r["summary"] for m, r in runs.items() if r["summary"].get("status", "ok") != "ok"}
    runs = {m: r for m, r in runs.items() if m not in pending}
    if not runs:
        return f"# 평가 리포트 {stamp}\n\n연결된 모드 결과가 없습니다."
    modes = list(runs)
    any_run = next(iter(runs.values()))
    k = any_run["summary"]["final_k"]
    lines = [f"# 평가 리포트 {stamp}", ""]
    lines.append(f"- 설정: {json.dumps(any_run.get('config', {}), ensure_ascii=False)}")
    lines.append("")

    # 요약 표
    ck = any_run["summary"]["candidate_k"]
    lines += ["Hit@1 = 1위 청크가 정답 조문인 문항 비율 (답변에 가장 크게 쓰이는 근거가 맞는지). "
              "정답이 여러 개인 문항은 하나만 맞아도 Hit이므로 Recall과 함께 본다.", "",
              f"| 모드 | Hit@1 | Hit@3 | Hit@{k} | Recall@{k} | MRR | Rerank 전 후보 Recall@{ck} | 평균 지연(초) | 평균 LLM 호출 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for mode, run in runs.items():
        s = run["summary"]
        cand = s.get(f"candidate_recall@{ck}", "-")
        lines.append(f"| {mode} | {s.get('hit@1', '-')} | {s.get('hit@3', '-')} | {s[f'hit@{k}']} | {s[f'recall@{k}']} "
                     f"| {s['mrr']} | {cand} | {s['avg_latency_sec']} | {s.get('avg_llm_calls', '-')} |")
    for mode, s in pending.items():
        lines.append(f"| {mode} | ⏸ {s.get('message', s.get('status'))} | | | | | | | |")
    lines.append("")

    # 유형별: 어떤 기법이 어떤 질문 유형에서 효과가 있었는지 (MQ=일상어, RR=경쟁 조항)
    types = sorted({t for run in runs.values() for t in run["summary"].get("by_type", {})})
    if types:
        first = next(iter(runs.values()))["summary"]["by_type"]
        lines += ["### 유형별 Hit@1 / MRR", "",
                  "| 모드 | " + " | ".join(f"{t} ({first.get(t, {}).get('n', '')}문항)" for t in types) + " |",
                  "|---|" + "---|" * len(types)]
        for mode, run in runs.items():
            by = run["summary"].get("by_type", {})
            cells = [f"{by[t].get('hit@1', '-')} / {by[t]['mrr']}" if t in by else "-" for t in types]
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
        lines.append("")

    answer_runs = {m: run["summary"]["answer"] for m, run in runs.items() if "answer" in run["summary"]}
    if answer_runs:
        lines += ["### 답변 지표", "",
                  "| 모드 | " + " | ".join(title for _, title in ANSWER_COLUMNS) + " | 평균 전체(초) |",
                  "|---|" + "---|" * (len(ANSWER_COLUMNS) + 1)]
        for mode, a in answer_runs.items():
            cells = [str(a.get(key, "-")) for key, _ in ANSWER_COLUMNS] + [str(a.get("avg_total_sec", "-"))]
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
        lines.append("")

    # 질문별 한눈에 보기
    rows_by_mode = {m: {r["id"]: r for r in run["rows"]} for m, run in runs.items()}
    compare = [m for m in modes if m != "baseline"] if "baseline" in modes else []
    lines += ["## 질문별 요약", "", "RR: 첫 정답 순위의 역수 (1위=1.00 → Hit@1, 2위=0.50, Top-K 밖=0.00). 판정은 baseline 대비.", "",
              "| ID | 유형 | " + " | ".join(f"{m} RR" for m in modes)
              + "".join(f" | {m} 판정" for m in compare) + " |",
              "|---|---|" + "---|" * (len(modes) + len(compare))]
    ids = list(rows_by_mode[modes[0]])
    for qid in ids:
        row0 = rows_by_mode[modes[0]][qid]
        rrs = [rows_by_mode[m][qid].get("rr") for m in modes]
        cells = ["-" if rr is None else f"{rr:.2f}" for rr in rrs]
        judges = [verdict(rows_by_mode["baseline"][qid].get("rr"), rows_by_mode[m][qid].get("rr")) for m in compare]
        lines.append(f"| {qid} | {row0.get('type')} | " + " | ".join(cells) + "".join(f" | {j}" for j in judges) + " |")
    lines.append("")

    # 질문별 상세
    lines += ["## 질문별 상세", ""]
    for qid in ids:
        row0 = rows_by_mode[modes[0]][qid]
        lines.append(f"### {qid} ({row0.get('type')}) {row0.get('question', '')}")
        gold = row0.get("gold", [])
        lines.append(f"- 정답: {', '.join(short(g) for g in gold) if gold else '(답변 불가 문항)'}")
        lines.append("")
        for mode in pending:
            lines.append(f"**{mode}** ⏸ 아직 연결되지 않았습니다")
            lines.append("")

        # 청크별 역할 요약: 어떤 청크가 어느 모드에서 검색·답변에 쓰였는지
        lines.append("| 청크 | 정답 | " + " | ".join(modes) + " |")
        lines.append("|---|---|" + "---|" * len(modes))
        for c in chunk_tables.get(qid, []):
            gold_mark = {True: "✅", False: "", None: "?"}[c["is_gold"]]
            missed = c["modes"].get("all")
            cells = [", ".join(c["modes"].get(m, [])) or ("검색 못 함" if missed else "-") for m in modes]
            lines.append(f"| {short(c['chunk_id'])} | {gold_mark} | " + " | ".join(cells) + " |")
        lines.append("")
        for mode in modes:
            row = rows_by_mode[mode][qid]
            rr = row.get("rr")
            lines.append(f"**{mode}** (RR={'-' if rr is None else f'{rr:.2f}'}, {row['latency_sec']}초)")
            lines.append("")
            lines.append("| 순위 | 청크 | 검색 점수 | Rerank 점수 | 정답 | 내용 |")
            lines.append("|---|---|---|---|---|---|")
            for r in row["retrieved"]:
                content = r["content"].replace("\n", " ").replace("|", "/")
                mark = "✅" if r["is_gold"] else "❌"
                rerank_score = "-" if r.get("rerank_score") is None else round(r["rerank_score"], 4)
                lines.append(f"| {r['rank']} | {short(r['chunk_id'])} | {r['score']} | {rerank_score} | {mark} | {content} |")
            lines.append("")

            if "candidate_ids" in row:
                gold_rank = row.get("candidate_gold_rank")
                where = f"{gold_rank}위" if gold_rank else "없음 (후보 누락)"
                lines.append(f"Rerank 전 후보 {len(row['candidate_ids'])}개 중 정답 위치: {where}, "
                             f"Rerank {row.get('rerank_sec', '-')}초")
                lines.append("")

            if row.get("per_query"):
                lines.append("확장 질의별 Dense 결과:")
                lines.append("")
                for i, pq in enumerate(row["per_query"]):
                    label = "원질문" if i == 0 else f"확장{i}"
                    top = ", ".join(short(c) for c in pq["chunk_ids"][:PER_QUERY_SHOW])
                    found = ""
                    if gold and "gold_rank" in pq:  # 이전 형식 결과에는 gold_rank가 없음
                        found = f" → 정답 {pq['gold_rank']}위" if pq.get("gold_rank") else " → 정답 없음"
                    lines.append(f"- {label}: {pq['query']}")
                    lines.append(f"  - 상위 {PER_QUERY_SHOW}: {top}{found}")
                lines.append("")

                change = mq_change(row["per_query"], row["retrieved"])
                added = ", ".join(f"{short(c)}{' ✅' if c in change['added_gold'] else ''}" for c in change["added"]) or "없음"
                lines.append(f"원질문만 썼을 때 Top-{len(change['original'])}: {', '.join(short(c) for c in change['original'])}  ")
                lines.append(f"→ 최종 Top-{len(change['final'])}: {', '.join(short(c) for c in change['final'])}  ")
                lines.append(f"→ 새로 들어온 청크: {added} / 밀려난 청크: {', '.join(short(c) for c in change['dropped']) or '없음'}")
                lines.append("")

            ans = row.get("answer")
            if ans:
                refuse = " (답변 불가로 응답)" if not ans["is_answerable"] else ""
                lines.append(f"답변{refuse} ({ans['generation_sec']}초):")
                lines.append("")
                lines.append("> " + ans["answer"].replace("\n", "\n> "))
                lines.append("")
                used = ", ".join(f"{short(c['chunk_id'])}(검색 {c['rank']}위) {'✅' if c['is_gold'] else '❌'}"
                                 for c in ans["used_chunks"]) or "없음"
                lines.append(f"답변이 실제 사용한 청크: {used}")
                lines.append("")
                judgement_lines = describe_judgement(ans)
                if judgement_lines:
                    lines += judgement_lines
                    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stamp", nargs="?", help="결과 파일 접두 (예: 20261008_120638)")
    args = parser.parse_args()

    stamp = args.stamp or latest_stamp()
    if stamp is None:
        sys.exit("실제 골든셋(golden_set.jsonl) 평가 결과가 없습니다. run_eval 을 먼저 실행하세요. "
                 "(더미 결과는 stamp를 지정)")
    runs = load_runs(stamp)
    if not runs:
        sys.exit(f"{stamp} 결과가 없습니다.")
    if not all("question" in r for run in runs.values() for r in run["rows"]):
        sys.exit("이전 형식 결과입니다. run_eval 을 다시 실행해 주세요.")

    out = RESULT_DIR / f"{stamp}_report.md"
    out.write_text(build_report(stamp, runs), encoding="utf-8")
    print(f"리포트: {out}")


if __name__ == "__main__":
    main()
