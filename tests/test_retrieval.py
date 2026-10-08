"""Dense boundary regressions with synthetic provenance and no external calls."""

from types import SimpleNamespace as NS

import pytest

from common.contracts import Chunk
from common.ingestion import embed_chunks
from common.retrieval import DenseRetriever, RetrievalError


class Embeddings:
    model = "fixture-embedding"

    def __init__(self):
        self.questions = []
        self.vector = [0.2, 0.4]

    def embed_documents(self, texts):
        return [self.vector for _ in texts]

    def embed_query(self, question):
        self.questions.append(question)
        return self.vector


def chunk(cid="a", **changes):
    return Chunk(chunk_id=cid, document_id="fixture", document_version="v1",
                 source_uri="fixture.pdf", content="합성 테스트 문장", article="제1조",
                 paragraph="①", page_start=2, page_end=3, **changes)


def point(cid="a", score=0.8, **changes):
    payload = chunk(cid).model_dump()
    payload.update(title="합성 자료", pdf_sha256="a" * 64, processed_scope="partial")
    payload.update(changes)
    return NS(payload=payload, score=score)


class Store:
    def __init__(self, points=()):
        self.points = list(points)
        self.calls = []
        self.size = 2

    def get_collection(self, **kwargs):
        self.calls.append(("info", kwargs))
        return NS(config=NS(params=NS(vectors=NS(size=self.size, distance="Cosine"))))

    def query_points(self, **kwargs):
        self.calls.append(("query", kwargs))
        return NS(points=self.points)


def retriever(points=(), **kwargs):
    embedding = kwargs.pop("embedding_model", Embeddings())
    store = kwargs.pop("qdrant_client", Store(points))
    record = kwargs.pop("embedding_record", {
        "provider": "monorouter", "model": "fixture-embedding", "dimensions": 2,
    })
    return DenseRetriever(collection="data_ingestion_fixture_v1", document_id="fixture",
                          document_version="v1", embedding_record=record,
                          embedding_model=embedding, qdrant_client=store, **kwargs)


def test_index_and_query_share_embedding_and_provenance():
    embedding = Embeddings()
    indexed = embed_chunks([chunk().model_dump()], provider="monorouter",
                           embedding_model=embedding)
    store = Store([point()])
    search = retriever(embedding_model=embedding, embedding_record=indexed.record,
                       qdrant_client=store)
    hits = search.retrieve("  테스트 질문  ", "dense", 1)
    assert hits[0].chunk == chunk()
    assert hits[0].rank == 1 and hits[0].retrieval_score == 0.8
    assert hits[0].rerank_score is None
    assert embedding.questions == ["테스트 질문"]
    args = store.calls[-1][1]
    assert args["query"] == list(indexed.vectors[0])
    assert args["collection_name"] == "data_ingestion_fixture_v1"
    assert args["limit"] == 1
    assert args["with_payload"] is True and args["with_vectors"] is False
    conditions = args["query_filter"].must
    assert {c.key: c.match.value for c in conditions} == {
        "document_id": "fixture", "document_version": "v1",
    }


def test_unique_ranked_top_k_and_explicit_candidates():
    search = retriever([point("b", 0.4), point("a", -0.2), point("b", 0.9),
                        point("c", 0.9)], candidate_count=8)
    hits = search.retrieve("질문", "dense", 2)
    assert [(h.chunk.chunk_id, h.rank, h.retrieval_score) for h in hits] == [
        ("b", 1, 0.9), ("c", 2, 0.9),
    ]
    assert search.qdrant_client.calls[-1][1]["limit"] == 8


def test_empty_results_are_not_external_failure():
    assert retriever().retrieve("질문", "dense", 3) == []


@pytest.mark.parametrize("question,mode,k", [
    ("", "dense", 1), ("   ", "dense", 1), ("a" * 2001, "dense", 1),
    (None, "dense", 1), ("질문", "unknown", 1), ("질문", "rerank", 1),
    ("질문", "multi_query", 1), ("질문", "multi_query_rerank", 1),
    ("질문", "dense", 0), ("질문", "dense", -1), ("질문", "dense", True),
    ("질문", "dense", 1.5),
])
def test_invalid_requests_never_call_dependencies(question, mode, k):
    search = retriever()
    with pytest.raises(ValueError):
        search.retrieve(question, mode, k)
    assert not search.qdrant_client.calls and not search.embedding_model.questions


@pytest.mark.parametrize("changes", [
    {"document_id": "other"}, {"document_version": "v2"},
    {"page_start": 0}, {"content": ""}, {"source_uri": None},
])
def test_invalid_or_out_of_scope_payload_fails_closed(changes):
    with pytest.raises(RetrievalError):
        retriever([point(**changes)]).retrieve("질문", "dense", 1)


def test_conflicting_duplicate_source_is_rejected():
    with pytest.raises(RetrievalError):
        retriever([point(), point(content="다른 원문")]).retrieve("질문", "dense", 1)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "0.8"])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(RetrievalError):
        retriever([point(score=score)]).retrieve("질문", "dense", 1)


@pytest.mark.parametrize("vector", [[], [1], [1, float("nan")], [True, 1], "bad"])
def test_query_vector_must_match_observed_index_dimensions(vector):
    embedding = Embeddings()
    embedding.vector = vector
    search = retriever(embedding_model=embedding)
    with pytest.raises(RetrievalError):
        search.retrieve("질문", "dense", 1)
    assert not any(name == "query" for name, _ in search.qdrant_client.calls)


def test_collection_dimension_mismatch_prevents_embedding_and_query():
    store = Store()
    store.size = 3
    search = retriever(qdrant_client=store)
    with pytest.raises(RetrievalError):
        search.retrieve("질문", "dense", 1)
    assert not search.embedding_model.questions


@pytest.mark.parametrize("record", [
    {"provider": "other", "model": "fixture-embedding", "dimensions": 2},
    {"provider": "monorouter", "model": "other", "dimensions": 2},
    {"provider": "monorouter", "model": "fixture-embedding", "dimensions": True},
])
def test_index_setting_mismatch_is_rejected(record):
    with pytest.raises(ValueError):
        retriever(embedding_record=record)


@pytest.mark.parametrize("stage", ["info", "embedding", "query"])
def test_external_failures_have_safe_messages(stage):
    search = retriever()
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic-sensitive-provider-detail")
    if stage == "info":
        search.qdrant_client.get_collection = fail
    elif stage == "embedding":
        search.embedding_model.embed_query = fail
    else:
        search.qdrant_client.query_points = fail
    with pytest.raises(RetrievalError) as caught:
        search.retrieve("질문", "dense", 1)
    assert "synthetic-sensitive" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("count", [0, True, 1.5])
def test_invalid_candidate_count(count):
    with pytest.raises(ValueError):
        retriever(candidate_count=count)


def test_candidates_cannot_be_less_than_k():
    search = retriever(candidate_count=1)
    with pytest.raises(ValueError):
        search.retrieve("질문", "dense", 2)
    assert not search.qdrant_client.calls


@pytest.mark.parametrize("vectors", [
    NS(size=2, distance="Dot"), {"named": NS(size=2, distance="Cosine")}, None,
])
def test_incompatible_collection_configuration_is_rejected(vectors):
    search = retriever()
    search.qdrant_client.get_collection = lambda **kwargs: NS(
        config=NS(params=NS(vectors=vectors)))
    with pytest.raises(RetrievalError):
        search.retrieve("질문", "dense", 1)
    assert not search.embedding_model.questions


@pytest.mark.parametrize("response", [NS(points=None), NS(points="bad"), None])
def test_malformed_search_response_is_not_empty_result(response):
    search = retriever()
    search.qdrant_client.query_points = lambda **kwargs: response
    with pytest.raises(RetrievalError):
        search.retrieve("질문", "dense", 1)


def test_missing_required_provenance_is_not_invented():
    candidate = point()
    del candidate.payload["page_end"]
    with pytest.raises(RetrievalError):
        retriever([candidate]).retrieve("질문", "dense", 1)


def test_unknown_article_and_paragraph_remain_unknown():
    hit = retriever([point(article=None, paragraph=None)]).retrieve("질문", "dense", 1)[0]
    assert hit.chunk.article is None and hit.chunk.paragraph is None


def test_model_configuration_drift_is_rejected_before_external_calls():
    search = retriever()
    search.embedding_model.model = "changed-model"
    with pytest.raises(RetrievalError):
        search.retrieve("질문", "dense", 1)
    assert not search.qdrant_client.calls


def test_default_factories_share_existing_configuration_without_network(monkeypatch):
    import sys
    embedding, store = Embeddings(), Store([point()])
    monkeypatch.setitem(sys.modules, "common.ai_model", NS(get_embedding_model=lambda: embedding))
    monkeypatch.setitem(sys.modules, "common.qdrant", NS(get_qdrant_client=lambda: store))
    search = DenseRetriever(collection="fixture_collection", document_id="fixture",
                            document_version="v1", embedding_record={
                                "provider": "monorouter", "model": embedding.model,
                                "dimensions": 2,
                            })
    assert search.retrieve("질문", "dense", 1)[0].chunk == chunk()


@pytest.mark.parametrize("field", ["collection", "document_id", "document_version"])
def test_scope_requires_explicit_nonblank_values(field):
    options = dict(collection="fixture_collection", document_id="fixture",
                   document_version="v1", embedding_record={
                       "provider": "monorouter", "model": "fixture-embedding", "dimensions": 2,
                   }, embedding_model=Embeddings(), qdrant_client=Store())
    options[field] = " "
    with pytest.raises(ValueError):
        DenseRetriever(**options)


def test_multi_query_includes_original_bounds_expansion_and_deduplicates():
    from common.retrieval import StrategyRetriever
    dense = retriever([point("a"), point("b", 0.5)])
    calls = []
    def expand(question, maximum):
        calls.append((question, maximum))
        return [question, "확장1", "확장1", "확장2", "확장3"]
    search = StrategyRetriever(dense, candidate_count=2, max_expansions=2,
                               total_candidate_budget=6, expand=expand)
    hits = search.retrieve(" 원질문 ", "multi_query", 2)
    assert dense.embedding_model.questions == ["원질문", "확장1", "확장2"]
    assert calls == [("원질문", 2)]
    assert [h.chunk.chunk_id for h in hits] == ["a", "b"]
    assert [h.rank for h in hits] == [1, 2]
    assert search.last_calls == {"expansion": 1, "embedding": 3, "collection": 3, "query": 3}


def test_rerank_uses_original_question_preserves_source_and_scores():
    from common.retrieval import StrategyRetriever
    dense = retriever([point("a", 0.9), point("b", 0.2)])
    seen = []
    def score(question, chunks):
        seen.append((question, chunks))
        return [0.1, 0.8]
    search = StrategyRetriever(dense, candidate_count=2, rerank=score)
    hits = search.retrieve("질문", "rerank", 1)
    assert seen[0][0] == "질문"
    assert hits[0].chunk == chunk("b")
    assert hits[0].retrieval_score == 0.2 and hits[0].rerank_score == 0.8


@pytest.mark.parametrize("scores", [[], [1], [1, float("nan")], [True, 1], "bad"])
def test_bad_reranker_output_fails_closed(scores):
    from common.retrieval import StrategyRetriever
    search = StrategyRetriever(retriever([point("a"), point("b")]), candidate_count=2,
                               rerank=lambda *args: scores)
    with pytest.raises(RetrievalError):
        search.retrieve("질문", "rerank", 1)


@pytest.mark.parametrize("mode", ["rerank", "multi_query", "multi_query_rerank"])
def test_unconfigured_strategies_never_silently_fall_back(mode):
    from common.retrieval import StrategyRetriever
    dense = retriever()
    with pytest.raises(ValueError):
        StrategyRetriever(dense, candidate_count=2).retrieve("질문", mode, 1)
    assert not dense.qdrant_client.calls


def test_total_candidate_budget_is_enforced_before_calls():
    from common.retrieval import StrategyRetriever
    dense = retriever()
    search = StrategyRetriever(dense, candidate_count=3, max_expansions=2,
                               total_candidate_budget=5, expand=lambda *args: ["확장"])
    with pytest.raises(ValueError):
        search.retrieve("질문", "multi_query", 1)
    assert not dense.qdrant_client.calls


def test_external_strategy_errors_are_safe():
    from common.retrieval import StrategyRetriever
    def fail(*args):
        raise RuntimeError("synthetic-sensitive-detail")
    search = StrategyRetriever(retriever([point()]), candidate_count=1, expand=fail)
    with pytest.raises(RetrievalError) as caught:
        search.retrieve("질문", "multi_query", 1)
    assert "synthetic-sensitive" not in str(caught.value)


def test_reranker_mutation_cannot_change_evidence():
    from common.retrieval import StrategyRetriever
    def mutate(question, chunks):
        chunks[0].content = "changed"
        return [1]
    search = StrategyRetriever(retriever([point()]), candidate_count=1, rerank=mutate)
    assert search.retrieve("질문", "rerank", 1)[0].chunk == chunk()


def test_multi_query_fuses_ranks_instead_of_comparing_query_scores():
    from common.retrieval import StrategyRetriever
    dense = retriever()
    dense.qdrant_client.query_points = lambda **kwargs: NS(points=(
        [point("a", 1), point("b", 0.5)] if dense.embedding_model.questions[-1] == "original"
        else [point("c", 100), point("b", 10)]))
    search = StrategyRetriever(dense, candidate_count=2, max_expansions=1,
                               expand=lambda *args: ["expansion"])
    hits = search.retrieve("original", "multi_query", 2)
    assert [h.chunk.chunk_id for h in hits] == ["b", "a"]
    assert hits[0].retrieval_score == 10  # preserved raw score, not a confidence


def test_cross_query_conflicting_chunk_ids_are_rejected():
    from common.retrieval import StrategyRetriever
    dense = retriever()
    dense.qdrant_client.query_points = lambda **kwargs: NS(points=[point(
        content=dense.embedding_model.questions[-1])])
    search = StrategyRetriever(dense, candidate_count=1, max_expansions=1,
                               expand=lambda *args: ["expansion"])
    with pytest.raises(RetrievalError):
        search.retrieve("original", "multi_query", 1)


@pytest.mark.parametrize("expanded", [None, "not a sequence", [""], [None], ["a" * 2001]])
def test_invalid_expansion_fails_before_dense_calls(expanded):
    from common.retrieval import StrategyRetriever
    dense = retriever()
    search = StrategyRetriever(dense, candidate_count=1, expand=lambda *args: expanded)
    with pytest.raises(RetrievalError):
        search.retrieve("question", "multi_query", 1)
    assert not dense.qdrant_client.calls


def test_empty_candidates_skip_reranker_and_zero_expansion_skips_expander():
    from common.retrieval import StrategyRetriever
    def unexpected(*args):
        raise AssertionError("must not call")
    search = StrategyRetriever(retriever(), candidate_count=1, max_expansions=0,
                               expand=unexpected, rerank=unexpected)
    assert search.retrieve("question", "multi_query_rerank", 1) == []
    assert search.last_calls == {"embedding": 1, "collection": 1, "query": 1}


def test_strategy_telemetry_resets_between_calls_and_counts_failed_requests():
    from common.retrieval import StrategyRetriever
    dense = retriever([point()])
    search = StrategyRetriever(dense, candidate_count=1)
    search.retrieve("q1", "dense", 1)
    def fail(*args, **kwargs):
        raise RuntimeError("private")
    dense.qdrant_client.get_collection = fail
    with pytest.raises(RetrievalError):
        search.retrieve("q2", "dense", 1)
    assert search.last_calls == {"collection": 1}
