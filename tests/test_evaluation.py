"""Synthetic metric/runner fixtures; these scores are not product results."""

import json
from types import SimpleNamespace as NS

import pytest

from common.contracts import AnswerResponse, Chunk, SearchHit
from common.evaluation import (
    EvaluationCase, EvaluationSet, RunSettings, compare, initial_dataset,
    retrieval_metrics,
)
from scripts.evaluate import main


def hit(cid, article="제1조", paragraph=None, **changes):
    fields = dict(chunk_id=cid, document_id="fixture", document_version="v1",
                  source_uri="fixture.pdf", content="synthetic", article=article,
                  paragraph=paragraph, page_start=1, page_end=1)
    fields.update(changes)
    return SearchHit(chunk=Chunk(**fields), rank=1, retrieval_score=0.5)


def dataset():
    return EvaluationSet(document_id="fixture", document_version="v1",
                         source_uri="fixture.pdf",
                         pdf_sha256="a" * 64, question_version="fixture-v1",
                         source_review="synthetic fixture, not legal ground truth",
                         cases=[
                             EvaluationCase(case_id="yes", question="question",
                                            expected=[{"article": "제1조"}, {"article": "제2조"}]),
                             EvaluationCase(case_id="no", question="out of scope",
                                            expected=[], unanswerable_reason="synthetic scope"),
                         ])


def settings():
    return RunSettings(chunking_version="fixture-v1", embedding_record={
        "provider": "fixture", "model": "fixture-model", "dimensions": 2},
        collection="fixture_collection", k=2, candidate_count=4,
        max_expansions=2, total_candidate_budget=12,
        answer_model="fixture-chat", prompt_version="fixture-v1",
        reranker_version="fixture-rerank", expansion_version="fixture-expand")


class Search:
    document_id = "fixture"
    document_version = "v1"
    collection = "fixture_collection"

    def __init__(self):
        self.run_settings = settings().model_dump()
        self.last_calls = {}

    def retrieve(self, question, mode, k):
        self.last_calls = {"embedding": 1, "query": 1}
        return [hit("a"), hit("b", "제3조").model_copy(update={"rank": 2})]


class Answer:
    def answer(self, question, hits):
        return AnswerResponse(status="insufficient_evidence", summary="not found")


def test_known_metrics_and_duplicate_chunks_do_not_inflate_recall():
    expected = [{"article": "제1조"}, {"article": "제2조"}]
    got = retrieval_metrics([hit("x", "제3조"), hit("a"), hit("a"), hit("b", "제2조")], expected, 3)
    assert got == {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 0.5}
    assert retrieval_metrics([], expected, 3) == {"hit_at_k": 0., "recall_at_k": 0., "mrr": 0.}


def test_paragraph_targets_and_multiple_chunks_of_one_article():
    expected = [{"article": "제1조", "paragraph": "②"}, {"article": "제2조"}]
    assert retrieval_metrics([hit("a", paragraph="①"), hit("b", paragraph="②")], expected, 2)["recall_at_k"] == 0.5
    assert retrieval_metrics([hit("a"), hit("b")], [{"article": "제1조"}], 2)["recall_at_k"] == 1


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_bad_metric_k(k):
    with pytest.raises(ValueError):
        retrieval_metrics([], [{"article": "제1조"}], k)


def test_unanswerable_has_no_retrieval_metric():
    assert retrieval_metrics([hit("a")], [], 2) is None


def test_same_conditions_three_modes_and_separate_unanswerable_counts():
    ticks = iter(range(100))
    report = compare(dataset(), settings(), retriever=Search(), answerer=Answer(), clock=lambda: next(ticks))
    assert set(report["modes"]) == {"dense", "rerank", "multi_query"}
    assert report["external_verified"] is False
    for result in report["modes"].values():
        assert result["metrics"] == {"hit_at_k": 1., "recall_at_k": 0.5, "mrr": 1.}
        assert result["unanswerable"] == {"total": 1, "evaluated": 1, "abstained": 1, "unsafe_answer": 0}
        assert result["records"][0]["elapsed_seconds"] == 1
        assert result["records"][0]["calls"] == {"embedding": 1, "query": 1, "answer_requests": 1}
        assert result["records"][0]["citations"] == []
        assert result["failures"] == ["yes"]  # incomplete recall / no grounded answer
    assert report["modes"]["rerank"]["delta_from_dense"]["recall_at_k"] == 0


def test_no_answerer_is_unchecked_not_abstention():
    report = compare(dataset(), settings(), retriever=Search())
    row = report["modes"]["dense"]
    assert row["unanswerable"]["evaluated"] == 0
    assert row["records"][0]["citations"] is None


def test_failure_details_are_redacted_and_failed_cases_are_retained():
    search = Search()
    def fail(*args):
        search.last_calls = {"query": 1}
        raise RuntimeError("synthetic-sensitive-value")
    search.retrieve = fail
    report = compare(dataset(), settings(), retriever=search)
    assert "synthetic-sensitive" not in json.dumps(report)
    for row in report["modes"].values():
        assert row["failures"] == ["yes", "no"]
        assert row["metrics"] is None
        assert row["retrieval_evaluated"] == 0
        assert row["records"][0]["calls"] == {"query": 1}


@pytest.mark.parametrize("field", ["collection", "document_id", "document_version", "run_settings"])
def test_run_scope_or_settings_drift_rejected(field):
    search = Search()
    setattr(search, field, "wrong")
    with pytest.raises(ValueError):
        compare(dataset(), settings(), retriever=search)


def test_returned_wrong_version_is_failure_not_score():
    search = Search()
    search.retrieve = lambda *args: [hit("a", document_version="v2")]
    report = compare(dataset(), settings(), retriever=search)
    assert report["modes"]["dense"]["metrics"] is None
    assert report["modes"]["dense"]["records"][0]["error"] == "evaluation_failed"


def test_returned_wrong_source_is_failure_not_score():
    search = Search()
    search.retrieve = lambda *args: [hit("a", source_uri="another-document.pdf")]
    report = compare(dataset(), settings(), retriever=search)
    for result in report["modes"].values():
        assert result["metrics"] is None
        assert result["records"][0]["error"] == "evaluation_failed"
        assert result["records"][0]["retrieved"] == []


def test_initial_set_is_versioned_eight_plus_two_with_pdf_provenance():
    data = initial_dataset()
    assert len(data.cases) == 10
    assert sum(bool(c.expected) for c in data.cases) == 8
    assert all(c.review_pages and c.review_note for c in data.cases)
    assert data.pdf_sha256 == "b5da7c6c499d0a2bb0537bccd665be715937193f61f09fd203c6189975094b63"
    assert data.source_uri.endswith(".pdf")
    assert data.fingerprint() == initial_dataset().fingerprint()


def test_dataset_rejects_duplicate_cases_and_unlabelled_impossible_question():
    data = dataset().model_dump()
    data["cases"].append(data["cases"][0])
    with pytest.raises(ValueError):
        EvaluationSet.model_validate(data)
    with pytest.raises(ValueError):
        EvaluationCase(case_id="bad", question="question", expected=[])


def test_cli_default_reports_unverified_without_scores(capsys):
    assert main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"] is None and report["external_verified"] is False
    assert len(report["dataset"]["cases"]) == 10


def test_cli_runs_injected_factory_and_writes_reviewable_report(tmp_path, capsys):
    d, s, out = (tmp_path / n for n in ["dataset.json", "settings.json", "report.json"])
    d.write_text(dataset().model_dump_json())
    s.write_text(settings().model_dump_json())
    assert main(["--run", "--dataset", str(d), "--settings", str(s), "--output", str(out)],
                factory=lambda config: (Search(), Answer())) == 0
    assert json.loads(out.read_text())["modes"]["dense"]["metrics"]["mrr"] == 1


def test_cli_does_not_echo_invalid_input_or_factory_errors(tmp_path, capsys):
    p = tmp_path / "invalid.json"
    p.write_text('synthetic-sensitive-value')
    assert main(["--dataset", str(p)]) == 1
    assert "synthetic-sensitive" not in capsys.readouterr().err


def test_failed_retrieval_is_counted_but_not_fabricated_as_measured_zero():
    search = Search()
    def retrieve(question, mode, k):
        if mode == "rerank":
            raise ValueError("provider unavailable")
        return [hit("a")]
    search.retrieve = retrieve
    result = compare(dataset(), settings(), retriever=search)
    assert result["modes"]["rerank"]["metrics"] is None
    assert result["modes"]["rerank"]["delta_from_dense"] is None
    assert result["modes"]["dense"]["retrieval_evaluated"] == 1


def test_actual_used_citations_are_preserved_and_invalid_citations_fail_closed():
    from common.contracts import Citation, Claim
    class Cited:
        def answer(self, question, hits):
            source = hits[0].chunk.model_dump(exclude={"content"})
            return AnswerResponse(status="answered", summary="synthetic",
                                  claims=[Claim(text="synthetic", citation_ids=["c1"])],
                                  citations=[Citation(**source, citation_id="c1", excerpt="synthetic")])
    result = compare(dataset(), settings(), retriever=Search(), answerer=Cited())
    row = result["modes"]["dense"]
    assert len(row["records"][0]["citations"]) == 1
    assert row["records"][0]["citations"][0]["chunk_id"] == "a"
    assert row["unanswerable"]["unsafe_answer"] == 1
    class Bad(Cited):
        def answer(self, question, hits):
            result = super().answer(question, hits)
            result.citations[0].excerpt = "absent"
            return result
    result = compare(dataset(), settings(), retriever=Search(), answerer=Bad())
    assert result["modes"]["dense"]["records"][0]["citations"] is None
    assert result["modes"]["dense"]["records"][0]["error"] == "evaluation_failed"


def test_settings_drift_during_run_cannot_be_scored():
    search = Search()
    def retrieve(*args):
        search.run_settings["k"] = 99
        return [hit("a")]
    search.retrieve = retrieve
    result = compare(dataset(), settings(), retriever=search)
    assert result["modes"]["dense"]["records"][0]["error"] == "evaluation_failed"
    assert result["modes"]["dense"]["metrics"] is None


def test_telemetry_labels_cannot_leak_provider_details():
    search = Search()
    def retrieve(*args):
        search.last_calls = {"synthetic-sensitive-value": 1}
        return [hit("a")]
    search.retrieve = retrieve
    result = compare(dataset(), settings(), retriever=search)
    assert "synthetic-sensitive" not in json.dumps(result)


def test_cli_external_failures_save_report_and_return_nonzero(tmp_path):
    d, s, out = (tmp_path / n for n in ["data.json", "settings.json", "report.json"])
    d.write_text(dataset().model_dump_json())
    s.write_text(settings().model_dump_json())
    search = Search()
    search.retrieve = lambda *args: (_ for _ in ()).throw(RuntimeError("private"))
    assert main(["--run", "--dataset", str(d), "--settings", str(s), "--output", str(out)],
                factory=lambda config: (search, None)) == 1
    assert json.loads(out.read_text())["modes"]["dense"]["metrics"] is None


def test_comparison_freezes_original_question_and_settings_against_adapter_mutation():
    search = Search()
    search.run_settings["embedding_record"]["provider"] = "other-provider"
    with pytest.raises(ValueError):
        compare(dataset(), settings(), retriever=search)
