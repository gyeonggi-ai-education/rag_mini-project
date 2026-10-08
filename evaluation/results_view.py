"""평가 결과 조회 (가장 최근 실행 또는 특정 실행)

FastAPI 등에서 그대로 반환할 수 있도록 JSON 직렬화 가능한 dict를 돌려준다.

    from evaluation.results_view import get_results, list_runs
    get_results()                   # 가장 최근 실행
    get_results("20261008_121436")  # 특정 실행

터미널 확인:
    uv run python -m evaluation.results_view            # 실제 골든셋 실행 중 가장 최근 요약
    uv run python -m evaluation.results_view --detail   # 질문별 검색 청크까지
    uv run python -m evaluation.results_view --list     # 실행 목록
    uv run python -m evaluation.results_view --question D03   # 문항 하나 + 청크 원문
"""

import argparse
import json
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "results"
MODE_ORDER = ["baseline", "rerank", "multi_query", "combined"]


def _result_files(stamp: str | None = None) -> list[Path]:
    pattern = f"{stamp}_*.json" if stamp else "2*_*.json"
    return sorted(RESULT_DIR.glob(pattern))


REAL_GOLDEN = "golden_set.jsonl"  # 실제 골든셋 10문항 (golden_set.md). 더미는 golden_set.dummy.jsonl


def _run_info(stamp: str) -> dict:
    """실행의 골든셋·데이터 버전 (결과 파일 하나의 config에서)"""
    for path in _result_files(stamp):
        try:
            config = json.loads(path.read_text(encoding="utf-8")).get("config", {})
        except (OSError, json.JSONDecodeError):
            continue
        if not config:  # 모드 미연결 등 config 없는 파일은 건너뜀
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows = result.get("rows") or []
        retrieved = rows[0].get("retrieved", []) if rows else []
        # 개발 초기 형식(검색 결과가 [chunk_id, article] 목록)은 현재 화면·API에서 읽을 수 없다
        compatible = not retrieved or isinstance(retrieved[0], dict)
        return {"golden_set": config.get("golden_set"), "data_version": config.get("data_version"),
                "collection": config.get("collection"), "compatible": compatible}
    return {"golden_set": None, "data_version": None, "collection": None, "compatible": False}


def list_runs(golden_set: str | None = None) -> list[dict]:
    """실행 목록 (최신순): 실행 시각, 모드, 골든셋, 데이터 버전. golden_set을 주면 그 골든셋 실행만"""
    runs: dict[str, list[str]] = {}
    for path in _result_files():
        stamp, mode = path.stem[:15], path.stem[16:]
        runs.setdefault(stamp, []).append(mode)
    listed = [{"stamp": s, "modes": runs[s], **_run_info(s)} for s in sorted(runs, reverse=True)]
    return [r for r in listed if golden_set is None or r["golden_set"] == golden_set]


def latest_stamp(golden_set: str | None = REAL_GOLDEN) -> str | None:
    """가장 최근 실행. 기본은 실제 골든셋 실행 중 최신 (더미 실행은 stamp를 직접 지정해서 본다)"""
    runs = [r for r in list_runs(golden_set) if r["compatible"]]
    return runs[0]["stamp"] if runs else None


def _hit_at(rows: list[dict], k: int) -> float:
    """저장된 검색 결과(retrieved[].is_gold)로 Hit@k 계산 (답변 불가 문항 제외)"""
    scored = [r for r in rows if r.get("answerable")]
    if not scored:
        return 0.0
    return round(sum(any(x["is_gold"] for x in r["retrieved"][:k]) for r in scored) / len(scored), 4)


def ensure_rank_metrics(run: dict) -> dict:
    """Hit@1·Hit@3이 없는 예전 결과 파일도 저장된 순위로 계산해 채운다 (API 호출 없음)"""
    summary, rows = run["summary"], run.get("rows") or []
    if summary.get("status", "ok") != "ok" or not rows:
        return run
    first = next((r["retrieved"][0] for r in rows if r.get("retrieved")), None)
    if not isinstance(first, dict):  # 개발 초기 형식은 계산하지 않음
        return run
    k = summary["final_k"]
    summary.setdefault("hit@1", _hit_at(rows, 1))
    if k > 3:
        summary.setdefault("hit@3", _hit_at(rows, 3))
    for type_name, by in summary.get("by_type", {}).items():
        by.setdefault("hit@1", _hit_at([r for r in rows if r.get("type") == type_name], 1))
    return run


def load_runs(stamp: str) -> dict[str, dict]:
    """{모드: 결과 파일 내용}, 모드 순서는 baseline → rerank → multi_query → combined"""
    runs = {}
    for path in _result_files(stamp):
        result = ensure_rank_metrics(json.loads(path.read_text(encoding="utf-8")))
        runs[result["summary"]["mode"]] = result
    order = {m: i for i, m in enumerate(MODE_ORDER)}
    return dict(sorted(runs.items(), key=lambda kv: order.get(kv[0], len(order))))


def verdict(base_rr: float | None, mode_rr: float | None) -> str | None:
    """baseline 대비 판정: 개선 / 악화 / 동일 (답변 불가 문항은 None)"""
    if base_rr is None or mode_rr is None:
        return None
    if mode_rr > base_rr:
        return "개선"
    if mode_rr < base_rr:
        return "악화"
    return "동일"


def mq_change(per_query: list[dict] | None, retrieved: list[dict]) -> dict | None:
    """Multi Query: 원질문만으로 검색했을 때의 Top-K와 최종 Top-K 비교

    original: 원질문 Dense Top-K (원래 쓰려던 청크)
    added: 확장 질의 덕분에 새로 들어온 청크 / dropped: 밀려난 청크
    """
    if not per_query:
        return None
    k = len(retrieved)
    original = per_query[0]["chunk_ids"][:k]
    final = [r["chunk_id"] for r in retrieved]
    gold = {r["chunk_id"] for r in retrieved if r["is_gold"]}
    return {
        "original": original,
        "final": final,
        "added": [c for c in final if c not in original],
        "added_gold": [c for c in final if c not in original and c in gold],
        "dropped": [c for c in original if c not in final],
    }


def chunk_roles(question: dict) -> list[dict]:
    """질문 하나에 등장한 청크별로 모드마다 어떤 역할이었는지 정리

    tags 예: "검색 2위", "답변 사용", "확장 질의로 추가", "밀려남", "Rerank 전 7위", "정답인데 검색 못 함"
    """
    gold_ids = {g for g in question.get("gold", []) if ":" in g}  # 청크 단위 정답 (더미)
    gold_articles = {g for g in question.get("gold", []) if ":" not in g}  # 조 번호 정답
    chunks: dict[str, dict] = {}

    def entry(chunk_id: str, article: str | None = None, is_gold: bool | None = None) -> dict:
        e = chunks.setdefault(chunk_id, {"chunk_id": chunk_id, "label": _short(chunk_id), "article": None,
                                         "is_gold": None, "modes": {}})
        if article and not e["article"]:
            e["article"] = article
        if is_gold is not None:
            e["is_gold"] = is_gold
        return e

    def tag(chunk_id: str, mode: str, text: str):
        entry(chunk_id)["modes"].setdefault(mode, []).append(text)

    for mode, m in question["modes"].items():
        if m.get("status", "ok") != "ok":
            continue
        for r in m["retrieved"]:
            entry(r["chunk_id"], r["article"], r["is_gold"])
            tag(r["chunk_id"], mode, f"검색 {r['rank']}위")
        candidate_ids = m.get("candidate_ids") or []
        for r in m["retrieved"]:
            if r["chunk_id"] in candidate_ids:
                tag(r["chunk_id"], mode, f"Rerank 전 {candidate_ids.index(r['chunk_id']) + 1}위")
        change = m.get("mq_change")
        if change:
            for c in change["added"]:
                tag(c, mode, "확장 질의로 추가")
            for c in change["dropped"]:
                tag(c, mode, "원질문 결과였으나 밀려남")
        if m.get("answer"):
            for c in m["answer"]["used_chunks"]:
                entry(c["chunk_id"], c["article"], c["is_gold"])
                tag(c["chunk_id"], mode, "답변 사용")

    for e in chunks.values():
        if e["is_gold"] is None and e["article"] and gold_articles:
            e["is_gold"] = e["article"] in gold_articles
        elif e["is_gold"] is None and gold_ids:
            e["is_gold"] = e["chunk_id"] in gold_ids
    for g in gold_ids - chunks.keys():
        e = entry(g, is_gold=True)
        e["modes"]["all"] = ["정답인데 어떤 모드도 검색 못 함"]

    # 정답 → 답변 사용 → 나머지 순
    def order(e):
        used = any("답변 사용" in tags for tags in e["modes"].values())
        return (not e["is_gold"], not used, e["chunk_id"])

    return sorted(chunks.values(), key=order)


def get_results(stamp: str | None = None) -> dict:
    """한 번의 평가 실행 결과를 모드 요약 + 질문별 비교로 묶어서 반환"""
    stamp = stamp or latest_stamp()
    if stamp is None:
        raise FileNotFoundError("실제 골든셋(golden_set.jsonl) 평가 결과가 없습니다. "
                                "evaluation.run_eval 을 먼저 실행하세요. (더미 결과는 stamp를 지정해서 조회)")
    runs = load_runs(stamp)
    if not runs:
        raise FileNotFoundError(f"{stamp} 실행 결과가 없습니다.")
    return build_results(stamp, runs)


def stable_metrics(summary: dict) -> dict | None:
    """K에 따라 이름이 바뀌는 지표(hit@5 등)를 고정 이름으로 (API·화면에서 쓰기 쉽게)"""
    if summary.get("status", "ok") != "ok":
        return None
    k, ck = summary["final_k"], summary["candidate_k"]
    return {
        "k": k,
        "hit_at_1": summary.get("hit@1"),
        "hit_at_3": summary.get("hit@3"),
        "hit": summary.get(f"hit@{k}"),
        "recall": summary.get(f"recall@{k}"),
        "mrr": summary.get("mrr"),
        "candidate_k": ck,
        "candidate_recall": summary.get(f"candidate_recall@{ck}"),
        "num_questions": summary.get("n"),
        "num_unanswerable": summary.get("num_unanswerable"),
        "avg_latency_sec": summary.get("avg_latency_sec"),
        "avg_llm_calls": summary.get("avg_llm_calls"),
        "monorouter_requests": summary.get("monorouter_requests"),
        "expansion_failures": summary.get("expansion_failures"),
        "total_tokens": summary.get("total_tokens"),
        "by_type": {
            t: {"n": v["n"], "hit_at_1": v.get("hit@1"), "hit": v.get(f"hit@{k}"),
                "recall": v.get(f"recall@{k}"), "mrr": v.get("mrr")}
            for t, v in summary.get("by_type", {}).items()
        },
        "answer": summary.get("answer"),
    }


def build_results(stamp: str, runs: dict[str, dict]) -> dict:
    """load_runs 결과를 API/리포트용 구조로 변환"""
    for run in runs.values():  # 화면 표시용 청크 이름 (해시 chunk_id → 조 번호)
        for row in run["rows"]:
            remember_labels(row.get("chunk_labels", {}))
            for r in row.get("retrieved", []):
                if isinstance(r, dict):
                    _LABELS.setdefault(r["chunk_id"], chunk_label(r))
    modes = list(runs)
    connected = {m: run for m, run in runs.items() if run["summary"].get("status", "ok") == "ok"}
    if not connected:
        raise FileNotFoundError(f"{stamp} 실행에 연결된 모드 결과가 없습니다.")
    first = next(iter(connected.values()))
    rows_by_mode = {m: {r["id"]: r for r in run["rows"]} for m, run in runs.items()}

    questions = []
    for row0 in first["rows"]:
        qid = row0["id"]
        base_rr = rows_by_mode["baseline"][qid].get("rr") if "baseline" in connected else None
        by_mode = {}
        for mode in modes:
            if mode not in connected:
                by_mode[mode] = {"status": runs[mode]["summary"]["status"], "message": runs[mode]["summary"].get("message")}
                continue
            row = rows_by_mode[mode].get(qid)
            if row is None:
                continue
            by_mode[mode] = {
                "status": "ok",
                "hit": row.get("hit"),
                "recall": row.get("recall"),
                "rr": row.get("rr"),
                "latency_sec": row.get("latency_sec"),
                "verdict_vs_baseline": verdict(base_rr, row.get("rr")) if mode != "baseline" else None,
                "retrieved": row.get("retrieved", []),
                "queries": row.get("queries"),
                "per_query": row.get("per_query"),
                "candidate_gold_rank": row.get("candidate_gold_rank"),
                "candidate_recall": row.get("candidate_recall"),
                "rerank_sec": row.get("rerank_sec"),
                "candidate_ids": row.get("candidate_ids"),
                "mq_change": mq_change(row.get("per_query"), row.get("retrieved", [])),
                # 답변 생성 결과 (--answer 로 실행한 경우): 답변, 실제 사용한 청크, 채점
                "answer": row.get("answer"),
            }
        question = {
            "id": qid,
            "type": row0.get("type"),
            "question": row0.get("question"),
            "answerable": row0.get("answerable"),
            "gold": row0.get("gold", []),
            "modes": by_mode,
        }
        question["chunks"] = chunk_roles(question)
        questions.append(question)

    config = dict(first.get("config", {}))
    config.pop("mode", None)
    return {
        "stamp": stamp,
        "config": config,
        "modes": modes,
        "summary": {m: run["summary"] for m, run in runs.items()},
        # 화면 표시용 고정 이름 지표 (미연결 모드는 status만)
        "metrics": {m: {"status": run["summary"].get("status", "ok"), "message": run["summary"].get("message"),
                        **(stable_metrics(run["summary"]) or {})}
                    for m, run in runs.items()},
        "questions": questions,
    }


def get_question(stamp: str | None, question_id: str, with_source: bool = False) -> dict:
    """질문 하나의 결과. with_source=True면 등장한 청크의 원문 전체를 원본(Qdrant)에서 불러온다."""
    data = get_results(stamp)
    question = next((q for q in data["questions"] if str(q["id"]) == str(question_id)), None)
    if question is None:
        raise FileNotFoundError(f"{data['stamp']} 실행에 {question_id} 문항이 없습니다.")
    if with_source:
        from evaluation.chunk_source import get_chunks  # Qdrant 연결은 이때만

        sources = {c["chunk_id"]: c for c in get_chunks([c["chunk_id"] for c in question["chunks"]],
                                                         collection=data["config"].get("collection"))}
        for c in question["chunks"]:
            src = sources.get(c["chunk_id"], {})
            c["source_found"] = src.get("found", False)
            c["content"] = src.get("content")
            c["article_title"] = src.get("article_title")
    return {"stamp": data["stamp"], "config": data["config"], **question}


def describe_judgement(answer: dict) -> list[str]:
    """답변 채점 결과를 사람이 읽는 문장으로 (터미널·리포트 공용)

    judgement: 규칙 기반(API 호출 없음) / llm_judgement: LLM 채점(--judge llm)
    구버전 결과(judgement에 LLM 점수가 들어 있던 형식)도 표시한다.
    """
    lines = []
    j = answer.get("judgement") or {}
    llm = answer.get("llm_judgement") or (j if "correctness" in j else None)
    if "refused_correctly" in j:
        return [f"채점: 답변 불가 처리 {'정상' if j['refused_correctly'] else '실패'}"]
    if "false_refusal" in j:
        return ["채점: 답변 가능한 문항을 거절함 (오거절)"]
    if "key_point_coverage" in j:
        coverage = "-" if j["key_point_coverage"] is None else f"{j['key_point_coverage']:.0%}"
        grounded = "-" if j["grounded_ratio"] is None else f"{j['grounded_ratio']:.0%}"
        lines.append(f"규칙 채점: 필수 요소 포함 {coverage}, 근거와 겹치는 문장 {grounded}")
        for p in j["missed_key_points"]:
            lines.append(f"  - 빠진 요소: {p}")
        for sentence in j["ungrounded_sentences"]:
            lines.append(f"  - 근거와 겹침 적은 문장: {sentence[:100]}")
    if llm:
        lines.append(f"LLM 채점: 정답성 {llm['correctness']} / 관련성 {llm['relevance']} / 충실성 {llm['faithfulness']}"
                     f" — {llm.get('reason', '')}")
        for claim in llm.get("unsupported_claims", []):
            lines.append(f"  - ⚠️ 근거 없는 주장: {claim}")
    return lines


ANSWER_COLUMNS = [  # (요약 키, 표 제목)
    ("used_gold_rate", "정답근거사용"),
    ("false_refusal_rate", "오거절"),
    ("unanswerable_refusal_rate", "불가거절"),
    ("avg_key_point_coverage", "필수요소포함"),
    ("avg_grounded_ratio", "근거겹침"),
    ("avg_correctness", "LLM정답성"),
    ("avg_faithfulness", "LLM충실성"),
    ("unsupported_claim_rate", "LLM근거없는주장"),
    ("avg_generation_sec", "생성(초)"),
]


_LABELS: dict[str, str] = {}  # chunk_id → 사람이 읽는 이름 (결과를 불러올 때 채움)


def chunk_label(chunk: dict) -> str:
    """청크 표시 이름: A의 chunk_id는 64자리 해시라서 조·항으로 표시한다

    "버전:이름" 형식(더미·임시 적재)은 이름 부분, 그 외는 조(+항), 정보가 없으면 ID 앞 8자리.
    """
    chunk_id = chunk.get("chunk_id", "")
    if ":" in chunk_id:
        return chunk_id.split(":", 1)[-1]
    article = chunk.get("article")
    if not article:
        return chunk_id[:8]
    paragraph = chunk.get("paragraph")
    return f"{article} {paragraph}항" if paragraph not in (None, "") else article


def remember_labels(labels: dict[str, str]):
    _LABELS.update(labels)


def _short(chunk_id: str) -> str:
    if chunk_id in _LABELS:
        return _LABELS[chunk_id]
    return chunk_id.split(":", 1)[-1] if ":" in chunk_id else chunk_id[:8]


def print_results(data: dict, detail: bool = False):
    k = data["config"].get("final_k") or next(iter(data["summary"].values()))["final_k"]
    print(f"=== 평가 결과 {data['stamp']} ===")
    print("설정:", json.dumps(data["config"], ensure_ascii=False))
    print()
    def cell(value):
        return "-" if value is None else value

    print(f"{'모드':<12} {'Hit@1':>7} {'Hit@3':>7} {'Hit@' + str(k):>7} {'Recall@' + str(k):>9} {'MRR':>6} "
          f"{'후보Recall':>10} {'지연(초)':>9}")
    for mode, s in data["summary"].items():
        if s.get("status", "ok") != "ok":
            print(f"{mode:<12} ⏸ {s.get('message', s.get('status'))}")
            continue
        cand = next((v for key, v in s.items() if key.startswith("candidate_recall@")), "-")
        print(f"{mode:<12} {cell(s.get('hit@1')):>7} {cell(s.get('hit@3')):>7} {s[f'hit@{k}']:>7} "
              f"{s[f'recall@{k}']:>9} {s['mrr']:>6} {cand:>10} {s['avg_latency_sec']:>9}")
    types = sorted({t for s in data["summary"].values() for t in s.get("by_type", {})})
    if types:
        print()
        print(f"{'유형별 Hit@1':<12} " + " ".join(f"{t + '(' + str(next((s['by_type'][t]['n'] for s in data['summary'].values() if t in s.get('by_type', {})), '')) + ')':>9}" for t in types))
        for mode, s in data["summary"].items():
            if s.get("status", "ok") == "ok":
                print(f"{mode:<12} " + " ".join(f"{cell(s['by_type'].get(t, {}).get('hit@1')):>9}" for t in types))
    answer_modes = {m: s["answer"] for m, s in data["summary"].items() if "answer" in s}
    if answer_modes:
        print()
        print(f"{'답변':<12} " + " ".join(f"{title:>10}" for _, title in ANSWER_COLUMNS))
        for mode, a in answer_modes.items():
            cells = ["-" if a.get(key) is None else str(a[key]) for key, _ in ANSWER_COLUMNS]
            print(f"{mode:<12} " + " ".join(f"{c:>10}" for c in cells))
    print()

    for q in data["questions"]:
        gold = ", ".join(_short(g) for g in q["gold"]) or "답변 불가"
        print(f"[{q['id']}] ({q['type']}) {q['question']}  | 정답: {gold}")
        for mode, m in q["modes"].items():
            if m.get("status", "ok") != "ok":
                print(f"    {mode:<12} ⏸ 아직 연결되지 않았습니다")
                continue
            rr = "-" if m["rr"] is None else f"{m['rr']:.2f}"
            judge = f" ({m['verdict_vs_baseline']})" if m["verdict_vs_baseline"] else ""
            top = " ".join(f"{_short(r['chunk_id'])}{'✅' if r['is_gold'] else '❌'}" for r in m["retrieved"])
            print(f"    {mode:<12} RR={rr}{judge:<6} 검색: {top}")
            change = m.get("mq_change")
            if change:
                added = " ".join(f"{_short(c)}{'✅' if c in change['added_gold'] else ''}" for c in change["added"]) or "없음"
                dropped = " ".join(_short(c) for c in change["dropped"]) or "없음"
                print(f"    {'':<12} 원질문 Top-K: {' '.join(_short(c) for c in change['original'])} → 추가: {added} / 밀려남: {dropped}")
            ans = m.get("answer")
            if ans:
                used = " ".join(f"{_short(c['chunk_id'])}{'✅' if c['is_gold'] else '❌'}" for c in ans["used_chunks"]) or "없음"
                refuse = "" if ans["is_answerable"] else " [답변 불가로 응답]"
                print(f"    {'':<12} 답변{refuse}: {ans['answer'][:120]}")
                print(f"    {'':<12} 답변이 사용한 청크: {used}")
                for line in describe_judgement(ans):
                    print(f"    {'':<12} {line}")
            if detail and m.get("per_query"):
                for i, pq in enumerate(m["per_query"]):
                    rank = pq.get("gold_rank")
                    found = ""
                    if q["answerable"] and "gold_rank" in pq:  # 답변 불가 문항은 정답이 없으므로 표시 안 함
                        found = f" → 정답 {rank}위" if rank else " → 정답 없음"
                    print(f"        {'원' if i == 0 else i}: {pq['query']}{found}")
        if detail and q.get("chunks"):
            print("    청크별 역할:")
            for c in q["chunks"]:
                mark = {True: "✅정답", False: "  "}.get(c["is_gold"], " ?")
                roles = " | ".join(f"{mode}: {', '.join(tags)}" for mode, tags in c["modes"].items())
                print(f"      {mark} {_short(c['chunk_id']):<14} {roles}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stamp", nargs="?", help="실행 시각 (생략 시 가장 최근)")
    parser.add_argument("--detail", action="store_true", help="확장 질의별 정답 순위까지 출력")
    parser.add_argument("--list", action="store_true", help="실행 목록 출력")
    parser.add_argument("--json", action="store_true", help="API 응답과 같은 JSON으로 출력")
    parser.add_argument("--question", help="문항 하나만 (원본 청크 원문까지 출력)")
    args = parser.parse_args()

    if args.list:
        for run in list_runs():
            kind = "실제" if run["golden_set"] == REAL_GOLDEN else "더미" if run["golden_set"] else "?"
            if not run["compatible"]:
                kind += ", 구형식"
            print(run["stamp"], f"[{kind}]", run["golden_set"], run["data_version"], ", ".join(run["modes"]))
        return

    try:
        _main(args)
    except FileNotFoundError as e:
        raise SystemExit(str(e))


def _main(args):
    if args.question:
        q = get_question(args.stamp, args.question, with_source=True)
        if args.json:
            print(json.dumps(q, ensure_ascii=False, indent=2))
            return
        print(f"[{q['id']}] {q['question']}  | 정답: {', '.join(_short(g) for g in q['gold']) or '답변 불가'}")
        for c in q["chunks"]:
            mark = {True: "✅정답", False: "", None: "?"}[c["is_gold"]]
            print(f"\n- {_short(c['chunk_id'])} {c.get('article_title') or ''} {mark}")
            for mode, tags in c["modes"].items():
                print(f"    {mode}: {', '.join(tags)}")
            print(f"    원문: {c['content'] if c['source_found'] else '(원본에서 찾지 못함)'}")
        return

    data = get_results(args.stamp)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print_results(data, detail=args.detail)


if __name__ == "__main__":
    main()
