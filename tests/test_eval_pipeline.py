"""4개 모드 평가 + 리포트 전체 흐름 확인 (가짜 Dense/LLM/Rerank, API 호출 없음)

실제 골든셋(조 번호 기준)과 실제 데이터와 비슷한 청크 ID 형식으로 확인한다.
실행: uv run python -m tests.test_eval_pipeline
"""

import sys
import types

import evaluation.run_eval as run_eval
from evaluation.rate_limit import limiter
import retrieval.multi_query as mq
from evaluation.report import build_report
from tests.test_multi_query import FakeLLM

limiter.rpm = 10**9  # 가짜 함수 테스트는 실제 API를 부르지 않으므로 호출 제한 해제

ARTICLES = ["제1조", "제2조", "제3조", "제7조", "제16조", "제17조", "제31조", "제32조", "제33조", "제34조", "제36조", "제43조"]


def chunk(article, paragraph=1):
    return {"chunk_id": f"law-v1:본문:{article}:{paragraph}항", "document_id": "law", "document_version": "v1",
            "article": article, "paragraph": paragraph, "is_supplementary": False, "content": f"{article} 본문"}


def fake_dense(question, k):
    # 질문 길이에 따라 순서를 돌려서 모드마다 다른 결과가 나오게 함
    shift = len(question) % len(ARTICLES)
    order = ARTICLES[shift:] + ARTICLES[:shift]
    return [{**chunk(a), "retrieval_score": 1 - i / 100, "rerank_score": None, "rank": i}
            for i, a in enumerate(order[:k], start=1)]


def fake_rerank(question, candidates, final_k):
    # 정답 조문을 아는 척하는 가짜: 조 번호 숫자가 큰 순서로 재정렬
    ranked = sorted(candidates, key=lambda c: -int(c["article"][1:-1]))[:final_k]
    return [{**c, "rerank_score": 1 - i / 10, "rank": i} for i, c in enumerate(ranked, start=1)]


def test_all_modes_and_report():
    sys.modules["retrieval.rerank"] = types.SimpleNamespace(rerank=fake_rerank)
    mq._default_llm = lambda **kw: FakeLLM(["확장 질의 하나", "조금 더 긴 확장 질의 두울"])

    golden = run_eval.load_golden(run_eval.DEFAULT_GOLDEN)
    runs = {m: run_eval.evaluate(m, golden, 10, 5, fake_dense) for m in run_eval.MODES}

    for mode, run in runs.items():
        s = run["summary"]
        assert s["n"] == 10 and 0 <= s["hit@5"] <= 1, mode
        assert s["hit@1"] <= s["hit@3"] <= s["hit@5"], mode   # Hit@1·Hit@3도 항상 기록
        if mode in ("rerank", "combined"):
            assert "candidate_recall@10" in s, mode
            assert all("candidate_ids" in r and r["retrieved"][0]["rerank_score"] is not None for r in run["rows"])
        if mode in ("multi_query", "combined"):
            pq = run["rows"][0]["per_query"]
            assert len(pq) == 3 and "gold_rank" in pq[0]
            assert run["summary"]["avg_llm_calls"] == 1

    report = build_report("test", runs)
    for text in ["rerank 판정", "combined 판정", "Rerank 전 후보", "확장 질의별 Dense 결과", "Rerank 점수"]:
        assert text in report, text


def fake_generate(question, chunks):
    # 첫 번째와 두 번째 청크를 근거로 썼다고 응답
    return {"answer": f"{chunks[0]['article']} 본문에 따르면 그렇다.", "is_answerable": True,
            "used_chunk_ids": [c["chunk_id"] for c in chunks[:2]]}


class Counter:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


class FakeBatchLLM:
    """질의 확장 묶음 요청용 가짜 LLM: 호출 횟수를 센다"""
    calls = 0

    def with_structured_output(self, schema, method=None):
        from langchain_core.runnables import RunnableLambda

        def run(prompt_value):
            FakeBatchLLM.calls += 1
            text = prompt_value.to_messages()[-1].content
            n = len([line for line in text.splitlines() if line.strip()])
            if schema.__name__ == "BatchExpandedQueries":
                return schema(items=[{"id": i, "queries": [f"확장 질의 {i}"]} for i in range(1, n + 1)])
            return schema(queries=["확장 질의 하나"])
        return RunnableLambda(run)


def _use_temp_caches(tmp_dir):
    """캐시를 임시 폴더로 (실제 evaluation/.cache 를 건드리지 않음)"""
    import evaluation.cache as cache_mod

    cache_mod.CACHE_DIR = tmp_dir
    run_eval._CACHES.clear()
    for name in ("embeddings", "expansions", "answers", "judgements", "reranks"):
        run_eval._CACHES[name] = cache_mod.JsonCache(name, enabled=True)


def test_rule_judge():
    from evaluation.answer_judge import rule_judge

    item = {"answerable": True, "key_points": ["생성형 인공지능", "과태료 3천만원"]}
    chunk = {"article": "제2조", "content": "생성형 인공지능이란 결과물을 생성하는 인공지능시스템을 말한다."}
    answer = {"answer": "생성형 인공지능이란 결과물을 생성하는 인공지능시스템입니다. 위반하면 징역 10년입니다.",
              "is_answerable": True}
    j = rule_judge(item, answer, [chunk])
    assert j["key_point_coverage"] == 0.5 and j["missed_key_points"] == ["과태료 3천만원"]
    assert j["grounded_ratio"] == 0.5 and "징역 10년" in j["ungrounded_sentences"][0]
    assert rule_judge({"answerable": False}, {"is_answerable": False}, [])["refused_correctly"] is True
    assert rule_judge(item, {"is_answerable": False, "answer": "어렵습니다"}, [])["false_refusal"] is True


def test_cached_embeddings_batch():
    import tempfile
    from pathlib import Path

    import evaluation.cache as cache_mod

    cache_mod.CACHE_DIR = Path(tempfile.mkdtemp())
    inner = types.SimpleNamespace(embed_documents=Counter(lambda texts: [[float(len(t))] for t in texts]),
                                  embed_query=Counter(lambda text: [float(len(text))]))
    emb = cache_mod.CachedEmbeddings(inner, cache_mod.JsonCache("embeddings"), "fake-model")
    emb.prefetch(["질문1", "질문2", "확장1", "확장2"])
    for text in ["질문1", "질문2", "확장1", "확장2"]:
        emb.embed_query(text)
    assert inner.embed_documents.calls == 1 and inner.embed_query.calls == 0  # 4개를 한 번에
    emb.embed_query("새 질문")
    assert inner.embed_query.calls == 1


def test_run_all_call_reduction():
    """prepare 묶음 확장, 답변 모드 선택, 문항당 1회 LLM 채점, Rerank 캐시"""
    import tempfile
    from pathlib import Path

    from langchain_core.runnables import RunnableLambda

    import evaluation.answer_judge as aj

    _use_temp_caches(Path(tempfile.mkdtemp()))
    FakeBatchLLM.calls = 0
    mq._default_llm = lambda **kw: FakeBatchLLM()
    rerank_counter = Counter(fake_rerank)
    sys.modules["retrieval.rerank"] = types.SimpleNamespace(rerank=rerank_counter)
    run_eval.rerank_available = lambda: True
    judge_calls = []

    def fake_judge(inputs):
        n = inputs["answers"].count("[답변 ")
        judge_calls.append(n)
        return aj.BatchJudgement(items=[{"id": i, "correctness": 4, "relevance": 5, "faithfulness": 3,
                                         "unsupported_claims": [], "reason": "테스트"} for i in range(1, n + 1)])
    aj._chain = RunnableLambda(fake_judge)

    golden = run_eval.load_golden(run_eval.DEFAULT_GOLDEN)[:4]
    golden.append({"id": "X1", "type": "답변불가", "question": "범위 밖 질문", "answerable": False,
                   "evidences": [], "key_points": []})
    generate = Counter(fake_generate)

    out = run_eval.run_all(golden, run_eval.MODES, fake_dense, 10, 5, generate,
                           answer_modes=["baseline", "multi_query"], judge="llm")
    results = out["results"]
    assert FakeBatchLLM.calls == 1, FakeBatchLLM.calls        # 5문항 확장을 한 번의 요청으로
    assert out["prepare"]["batch_expanded"] == 5
    assert all("answer" not in r for r in results["rerank"]["rows"] + results["combined"]["rows"])
    assert generate.calls <= 10                               # 답변은 2개 모드만 (같은 청크면 재사용)
    assert len(judge_calls) == 4                              # 답변 가능 4문항 × 1회 (모드 묶음)
    assert results["baseline"]["rows"][0]["answer"]["llm_judgement"]["correctness"] == 4
    assert "_used_full" not in results["baseline"]["rows"][0]["answer"]
    assert results["baseline"]["summary"]["answer"]["avg_correctness"] == 4
    assert "avg_key_point_coverage" in results["baseline"]["summary"]["answer"]
    rerank_calls = rerank_counter.calls

    # 같은 평가 재실행: 확장·Rerank·답변·채점 모두 캐시
    generate.calls = 0
    judge_calls.clear()
    run_eval.run_all(golden, run_eval.MODES, fake_dense, 10, 5, generate,
                     answer_modes=["baseline", "multi_query"], judge="llm")
    assert FakeBatchLLM.calls == 1 and rerank_counter.calls == rerank_calls
    assert generate.calls == 0 and judge_calls == []

    report = build_report("test", results)
    for text in ["답변 지표", "답변이 실제 사용한 청크", "규칙 채점", "LLM 채점", "새로 들어온 청크", "답변 불가 처리 실패"]:
        assert text in report, text

    from evaluation.results_view import print_results  # 콘솔 출력 경로도 에러 없이 도는지
    import io, contextlib
    from evaluation.results_view import build_results
    with contextlib.redirect_stdout(io.StringIO()):
        print_results(build_results("test", results), detail=True)
    run_eval._CACHES.clear()


def test_trace_question():
    import evaluation.run_eval as re_
    import evaluation.trace as tr

    sys.modules["retrieval.rerank"] = types.SimpleNamespace(rerank=fake_rerank)
    mq._default_llm = lambda **kw: FakeLLM(["확장 질의 하나"])
    tr.get_dense_fn = lambda collection: fake_dense
    tr.rerank_available = lambda: True
    t = tr.trace_question("고영향 인공지능이란?", final_k=3, gold_articles=["제34조"])
    t["modes"]["baseline"]["answer"] = None
    assert set(t["modes"]) == {"baseline", "rerank", "multi_query", "combined"}
    assert t["modes"]["multi_query"]["mq_change"]["original"]
    assert t["modes"]["rerank"]["retrieved"][0]["rerank_score"] is not None
    md = tr.trace_to_markdown(t)
    assert "확장 질의와 질의별 Dense 결과" in md and "원질문 결과 대비 변화" in md

    tr.rerank_available = lambda: False
    t = tr.trace_question("질문", modes=["baseline", "rerank"], final_k=3)
    assert t["modes"]["rerank"]["status"] == "not_connected"
    assert "⏸" in tr.trace_to_markdown(t)


def test_hash_chunk_ids_shown_as_articles():
    """A의 chunk_id는 64자리 해시 → 화면·리포트에는 조 번호로 표시"""
    import hashlib

    from evaluation.results_view import _short, build_results

    def hash_dense(question, k):
        rows = fake_dense(question, k)
        return [{**r, "chunk_id": hashlib.sha256(r["article"].encode()).hexdigest(), "paragraph": None}
                for r in rows]

    run_eval._CACHES.clear()
    mq._default_llm = lambda **kw: FakeLLM(["확장 질의 하나"])
    golden = run_eval.load_golden(run_eval.DEFAULT_GOLDEN)[:2]
    runs = {m: run_eval.evaluate(m, golden, 10, 3, hash_dense, None, None) for m in ["baseline", "multi_query"]}
    data = build_results("test", runs)
    some_id = runs["baseline"]["rows"][0]["retrieved"][0]["chunk_id"]
    assert len(some_id) == 64 and _short(some_id).startswith("제")
    assert all(c["label"].startswith("제") for q in data["questions"] for c in q["chunks"])
    report = build_report("test", runs)
    assert some_id not in report and some_id[:8] not in report


if __name__ == "__main__":
    test_all_modes_and_report()
    print("PASS test_all_modes_and_report")
    test_rule_judge()
    print("PASS test_rule_judge")
    test_cached_embeddings_batch()
    print("PASS test_cached_embeddings_batch")
    test_run_all_call_reduction()
    print("PASS test_run_all_call_reduction")
    test_trace_question()
    print("PASS test_trace_question")
    test_hash_chunk_ids_shown_as_articles()
    print("PASS test_hash_chunk_ids_shown_as_articles")
