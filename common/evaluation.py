"""Versioned evaluation, not legal accuracy certification.

Metrics use unique relevant article/paragraph/item targets, not chunk counts.
No external run is implied by local fixture verification. Adapter request counts
exclude retries inside adapters. Answer faithfulness requires human review.
"""

import hashlib
import json
import time
from typing import Annotated

from pydantic import Field, model_validator

from common.contracts import (
    AnswerResponse, AskRequest, Contract, NonBlank, Page, SearchHit,
)


class EvidenceTarget(Contract):
    article: NonBlank
    paragraph: NonBlank | None = None
    item: NonBlank | None = None

    def matches(self, chunk):
        return all(value is None or getattr(chunk, key) == value
                   for key, value in self.model_dump().items())


class EvaluationCase(Contract):
    case_id: NonBlank
    question: NonBlank
    expected: list[EvidenceTarget]
    unanswerable_reason: NonBlank | None = None
    review_pages: list[Page] = Field(default_factory=list)
    review_note: NonBlank | None = None

    @model_validator(mode="after")
    def check_case(self):
        self.question = AskRequest(question=self.question).question
        if bool(self.expected) == bool(self.unanswerable_reason):
            raise ValueError("Expected evidence or an explicit unanswerable reason")
        targets = [t.model_dump_json() for t in self.expected]
        if len(set(targets)) != len(targets):
            raise ValueError("Duplicate evidence targets")
        return self


class EvaluationSet(Contract):
    document_id: NonBlank
    document_version: NonBlank
    source_uri: NonBlank
    pdf_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    question_version: NonBlank
    source_review: NonBlank
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def check_ids(self):
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate case IDs")
        return self

    def fingerprint(self):
        return _fingerprint(self.model_dump())


class EmbeddingRecord(Contract):
    provider: NonBlank
    model: NonBlank
    dimensions: Page


class RunSettings(Contract):
    chunking_version: NonBlank
    embedding_record: EmbeddingRecord
    collection: NonBlank
    k: Page
    candidate_count: Page
    max_expansions: Annotated[int, Field(strict=True, ge=0)]
    total_candidate_budget: Page
    answer_model: NonBlank
    prompt_version: NonBlank
    reranker_version: NonBlank
    expansion_version: NonBlank

    @model_validator(mode="after")
    def check_budget(self):
        if self.candidate_count < self.k:
            raise ValueError("Candidate count is below K")
        if self.total_candidate_budget < self.candidate_count * (1 + self.max_expansions):
            raise ValueError("Total budget is below configured query slots")
        return self


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def retrieval_metrics(hits, expected, k):
    """Hit@K, unique target Recall@K and truncated MRR (zero outside K).

    Input list order is the ranking. Repeated chunk IDs are counted once before
    truncation. Unanswerable cases have no retrieval metric, even if hits exist.
    """
    if type(k) is not int or k < 1:
        raise ValueError("Expected positive integer K")
    targets = [EvidenceTarget.model_validate(t.model_dump() if isinstance(t, EvidenceTarget) else t)
               for t in expected]
    if not targets:
        return None
    if len({t.model_dump_json() for t in targets}) != len(targets):
        raise ValueError("Duplicate evidence targets")
    unique = {}
    for hit in hits:
        checked = SearchHit.model_validate(hit.model_dump())
        cid = checked.chunk.chunk_id
        if cid in unique and unique[cid].chunk != checked.chunk:
            raise ValueError("Conflicting chunk ID")
        unique.setdefault(cid, checked)
    chunks = [h.chunk for h in list(unique.values())[:k]]
    matched = {i for i, target in enumerate(targets)
               if any(target.matches(chunk) for chunk in chunks)}
    first = next((rank for rank, chunk in enumerate(chunks, 1)
                  if any(target.matches(chunk) for target in targets)), None)
    return {"hit_at_k": float(bool(matched)), "recall_at_k": len(matched) / len(targets),
            "mrr": 1 / first if first else 0.0}


def _validate_run(dataset, settings, retriever):
    for field in ("document_id", "document_version"):
        if getattr(retriever, field, None) != getattr(dataset, field):
            raise ValueError("Evaluation document scope differs from retriever")
    if getattr(retriever, "collection", None) != settings.collection:
        raise ValueError("Evaluation collection differs from retriever")
    # Factory must bind all experimental labels to its actual adapters. Checking
    # labels cannot independently verify a remote provider or index contents.
    if getattr(retriever, "run_settings", None) != settings.model_dump():
        raise ValueError("Adapter must declare matching run_settings")
    dense = getattr(retriever, "dense", None)
    if dense is not None:
        if (dense.indexed_model != settings.embedding_record.model
                or dense.provider != settings.embedding_record.provider
                or dense.dimensions != settings.embedding_record.dimensions
                or retriever.candidate_count != settings.candidate_count
                or retriever.max_expansions != settings.max_expansions
                or retriever.total_candidate_budget != settings.total_candidate_budget):
            raise ValueError("Actual retrieval settings differ from evaluation")


def _calls(retriever):
    result = dict(getattr(retriever, "last_calls", {}))
    labels = {"embedding", "collection", "query", "expansion", "rerank", "retrieval"}
    if any(key not in labels or type(value) is not int or value < 0
           for key, value in result.items()):
        raise ValueError("Invalid request counts")
    return result


def compare(dataset: EvaluationSet, settings: RunSettings, *, retriever,
            answerer=None, clock=time.perf_counter):
    """Sequential comparison with fixed inputs and separately counted failures.

    Exceptions are redacted. Failure after successful retrieval keeps retrieval
    metrics, while invalid retrieval has no measured metric. Means cover valid
    retrievals only; deltas require complete retrieval coverage in both modes.
    No answerer means abstention and used evidence are unverified (None), never a
    successful refusal. Citations are structurally checked against exact hits.
    This does not assess whether a claim semantically follows from its excerpt.
    """
    dataset = EvaluationSet.model_validate(dataset.model_dump())
    settings = RunSettings.model_validate(settings.model_dump())
    _validate_run(dataset, settings, retriever)
    report = {"dataset": dataset.model_dump(), "dataset_sha256": dataset.fingerprint(),
              "settings": settings.model_dump(), "settings_sha256": _fingerprint(settings.model_dump()),
              "external_verified": False,
              "verification_note": "Adapter execution only; provider support, PDF semantics and answer faithfulness require separate review.",
              "call_count_unit": "adapter requests; internal retries and HTTP calls excluded",
              "modes": {}}
    for mode in ("dense", "rerank", "multi_query"):
        records, scores, failures = [], [], []
        unavailable = {"total": sum(not c.expected for c in dataset.cases),
                       "evaluated": 0, "abstained": 0, "unsafe_answer": 0}
        for case in dataset.cases:
            start = clock()
            metric = None
            row = {"case_id": case.case_id, "question": case.question,
                   "retrieved": [], "citations": None, "answer_status": None,
                   "error": None, "calls": {}}
            try:
                _validate_run(dataset, settings, retriever)
                hits = [SearchHit.model_validate(h.model_dump())
                        for h in retriever.retrieve(case.question, mode, settings.k)]
                if len(hits) > settings.k or len({h.chunk.chunk_id for h in hits}) != len(hits):
                    raise ValueError("Invalid final result size or duplicate IDs")
                for rank, h in enumerate(hits, 1):
                    if (h.rank != rank or h.chunk.document_id != dataset.document_id
                            or h.chunk.document_version != dataset.document_version
                            or h.chunk.source_uri != dataset.source_uri):
                        raise ValueError("Invalid scope or rank")
                _validate_run(dataset, settings, retriever)
                row["calls"] = _calls(retriever)
                # Snapshot context before invoking any answer adapter.
                original_hits = [h.model_copy(deep=True) for h in hits]
                metric = retrieval_metrics(original_hits, case.expected, settings.k)
                row["retrieved"] = [h.model_dump() for h in original_hits]
                if answerer is not None:
                    row["calls"]["answer_requests"] = 1
                    answer = answerer.answer(case.question, hits)
                    answer = AnswerResponse.model_validate(answer.model_dump())
                    answer.validate_context([h.chunk for h in original_hits])
                    _validate_run(dataset, settings, retriever)
                    row["answer_status"] = answer.status
                    row["citations"] = [c.model_dump() for c in answer.citations]
                    row["answer"] = answer.model_dump()
                    if not case.expected:
                        unavailable["evaluated"] += 1
                        abstained = answer.status in {"insufficient_evidence", "needs_clarification"}
                        unavailable["abstained" if abstained else "unsafe_answer"] += 1
                _validate_run(dataset, settings, retriever)
            except Exception:
                row["error"] = "evaluation_failed"
                try:
                    _validate_run(dataset, settings, retriever)
                except Exception:
                    metric = None
                    row["retrieved"], row["citations"], row["answer_status"] = [], None, None
                    row.pop("answer", None)
                # Preserve safe counts even on a failing adapter request.
                try:
                    row["calls"] = _calls(retriever) | {
                        k: v for k, v in row["calls"].items() if k == "answer_requests"}
                except Exception:
                    row["calls"] = None
            row["elapsed_seconds"] = max(0.0, clock() - start)
            row["metrics"] = metric
            if metric is not None:
                scores.append(metric)
            if (row["error"] or (metric is not None and metric["recall_at_k"] < 1)
                    or (case.expected and row["answer_status"] in {"insufficient_evidence", "needs_clarification"})
                    or (not case.expected and row["answer_status"] in {"answered", "partial"})):
                failures.append(case.case_id)
            records.append(row)
        mean = {key: sum(s[key] for s in scores) / len(scores)
                for key in ("hit_at_k", "recall_at_k", "mrr")} if scores else None
        report["modes"][mode] = {"metrics": mean,
                                 "answerable_total": sum(bool(c.expected) for c in dataset.cases),
                                 "retrieval_evaluated": len(scores),
                                 "unanswerable": unavailable, "records": records,
                                 "failures": failures}
    dense_result = report["modes"]["dense"]
    baseline = dense_result["metrics"]
    for result in report["modes"].values():
        result["delta_from_dense"] = {key: result["metrics"][key] - baseline[key]
                                      for key in baseline} if (
            baseline is not None and result["metrics"] is not None
            and result["retrieval_evaluated"] == result["answerable_total"]
            and dense_result["retrieval_evaluated"] == dense_result["answerable_total"]
        ) else None
    return report


def initial_dataset() -> EvaluationSet:
    """Eight answerable and two out-of-scope questions for the exact local PDF.

    Reviewed with Poppler text extraction on 2026-10-08. Article relevance labels
    are a small starting set, not legal answers or exhaustive annotation. Pages
    are physical PDF pages. No inferred paragraphs are assigned to whole-article
    chunks; narrow paragraph targets may be used for future structured indexes.
    """
    specs = [
        ("q01", "이 법의 목적은 무엇인가요?", "제1조", [1], "목적 조문 전체와 대조"),
        ("q02", "영향받는 자에게 제공되는 설명은 어떤 범위인가요?", "제3조", [2], "제3조 제2항의 기술적ㆍ합리적으로 가능한 범위와 대조"),
        ("q03", "국외 행위와 국방·국가안보 목적 인공지능에 대한 적용범위는 어떻게 정해져 있나요?", "제4조", [2], "제1항 적용 조건과 제2항 제외 조건을 함께 대조"),
        ("q04", "인공지능 기본계획은 누가 몇 년마다 수립하고 어떤 심의를 거치나요?", "제6조", [2, 3], "제1항의 주체·3년 주기·심의 및 경미한 변경 예외와 대조"),
        ("q05", "생성형 인공지능 제품·서비스의 사전 고지와 결과물 표시는 어떻게 규정되어 있나요?", "제31조", [12], "제1항 사전 고지와 제2항 결과물 표시를 대조"),
        ("q06", "누적 연산량 기준에 해당하는 인공지능시스템의 안전성 확보 사항과 이행 결과 제출은 무엇인가요?", "제32조", [12, 13], "페이지 경계의 제1항 두 호와 제2항 제출을 대조; 연산량 수치를 추정하지 않음"),
        ("q07", "고영향 인공지능 해당 여부는 언제 검토하며 누구에게 확인을 요청할 수 있나요?", "제33조", [13], "제1항 사전 검토와 장관 확인 요청을 대조"),
        ("q08", "고영향 인공지능 사업자의 안전성·신뢰성 조치에는 어떤 내용이 포함되나요?", "제34조", [13], "제1항 각 호의 위험관리·설명·보호·사람의 감독·문서 작성 및 보관 등과 대조"),
    ]
    cases = [EvaluationCase(case_id=cid, question=q, expected=[EvidenceTarget(article=article)],
                            review_pages=pages, review_note=note)
             for cid, q, article, pages, note in specs]
    for cid, q, reason in [
        ("q09", "GPT-4 서비스의 월 구독료는 얼마인가요?", "이 PDF는 특정 상용 서비스의 요금표를 제공하지 않음"),
        ("q10", "오늘 서울의 날씨와 강수 확률은 얼마인가요?", "이 PDF는 실시간 기상 자료를 제공하지 않음"),
    ]:
        cases.append(EvaluationCase(case_id=cid, question=q, expected=[],
                                    unanswerable_reason=reason, review_pages=list(range(1, 16)),
                                    review_note="15페이지 전체 추출 텍스트의 문서 범위 대조; 문서 밖 질문"))
    return EvaluationSet(document_id="law-21311",
                         document_version="[시행 2026. 7. 21.] [법률 제21311호, 2026. 1. 20., 일부개정]",
                         source_uri="file:///Users/steve.joo/Downloads/인공지능 발전과 신뢰 기반 조성 등에 관한 기본법(법률)(제21311호)(20260721).pdf",
                         pdf_sha256="b5da7c6c499d0a2bb0537bccd665be715937193f61f09fd203c6189975094b63",
                         question_version="law-21311-8plus2-v1",
                         source_review="2026-10-08: local PDF SHA-256 matched ingestion manifest; full Poppler text scope review and relevant article/page comparison; rendered pages 1, 2, 3, 12, 13 visually checked. Small relevance set, no semantic answer certification.",
                         cases=cases)
