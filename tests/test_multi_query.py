"""가짜 Dense/LLM으로 Multi Query 병합 로직과 지표를 확인 (연결 확인용, 성능 수치 아님)

실행: uv run python -m tests.test_multi_query
"""

from evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank
from retrieval.multi_query import expand_queries, multi_query_retrieve


def chunk(article):
    return {"chunk_id": f"v0:{article}", "document_id": "fake", "document_version": "fake",
            "article": article, "content": "", "retrieval_score": 0.0, "rerank_score": None, "rank": 0}


FAKE_INDEX = {
    "원질문": ["제1조", "제2조", "제3조"],
    "확장1": ["제2조", "제4조"],
    "확장2": ["제2조", "제1조"],
}


def fake_dense(query, k):
    return [chunk(a) for a in FAKE_INDEX.get(query, [])][:k]


class FakeLLM:
    def __init__(self, queries=None, fail=False):
        self.queries, self.fail = queries, fail

    def with_structured_output(self, schema, method=None):
        from langchain_core.runnables import RunnableLambda

        def run(_):
            if self.fail:
                raise RuntimeError("model down")
            return schema(queries=self.queries)
        return RunnableLambda(run)


def test_expand_keeps_original_and_dedups():
    qs = expand_queries("원질문", llm=FakeLLM(["확장1", "원질문", " 확장1 ", "확장2", "확장3"]))
    assert qs == ["원질문", "확장1", "확장2", "확장3"]


def test_expand_fallback_on_failure():
    assert expand_queries("원질문", llm=FakeLLM(fail=True)) == ["원질문"]


def test_rrf_merge_and_dedup():
    stats = {}
    results = multi_query_retrieve("원질문", candidate_k=3, dense_fn=fake_dense,
                                   llm=FakeLLM(["확장1", "확장2"]), stats=stats)
    assert [r["article"] for r in results] == ["제2조", "제1조", "제4조"]
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert stats["dense_calls"] == 3 and stats["queries"][0] == "원질문"


def test_dense_many_fn_called_once():
    calls = []

    def fake_dense_many(queries, k):
        calls.append(list(queries))
        return [fake_dense(q, k) for q in queries]

    stats = {}
    results = multi_query_retrieve("원질문", candidate_k=3, dense_many_fn=fake_dense_many,
                                   llm=FakeLLM(["확장1", "확장2"]), stats=stats)
    assert len(calls) == 1 and calls[0] == ["원질문", "확장1", "확장2"]
    assert [r["article"] for r in results] == ["제2조", "제1조", "제4조"]   # dense_fn과 같은 결과
    assert stats["dense_calls"] == 1


def test_metrics():
    results = [chunk("제1조"), chunk("제2조")]
    gold = [{"article": "제2조", "paragraph": None}, {"article": "제9조", "paragraph": None}]
    assert hit_at_k(results, gold, 2) == 1.0
    assert recall_at_k(results, gold, 2) == 0.5
    assert reciprocal_rank(results, gold, 2) == 0.5
    assert hit_at_k(results, gold, 1) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
