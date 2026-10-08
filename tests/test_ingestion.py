"""Synthetic text only; these fixtures make no statements about actual law."""

import hashlib
import json
from dataclasses import replace

import pytest

from common.contracts import Chunk, Document
from common.ingestion import (
    ExtractedDocument, PdfPage, SourceManifest, chunk_document, prepare_pdf, ingest_pdf,
)
from scripts.ingest import main


def fixture(tmp_path, texts, version="fixture-v1"):
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"synthetic PDF bytes; extractor is injected")
    document = Document(
        document_id="synthetic", document_version=version, title="Synthetic",
        source_uri=path.as_uri(), pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        processed_scope="full",
    )
    return path, document


def test_multi_page_paragraph_items_and_exceptions_remain_together(tmp_path):
    texts = ["제1조(시험) ① 조건이 맞으면 다음을 적용한다.\n1. 첫 항목\n",
             "2. 두 번째 항목\n다만, 예외 조건이면 제외한다.\n② 다른 조건이다.\n제2조(다음) 다음 내용"]
    path, doc = fixture(tmp_path, texts)
    calls = []
    chunks = prepare_pdf(path, doc, page_count=2,
                         extractor=lambda p: calls.append(p) or texts)
    assert calls == [path]
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    first = chunks[0]
    assert (first.page_start, first.page_end) == (1, 2)
    assert all(t in first.content for t in ["①", "②", "1. 첫 항목", "2. 두 번째 항목", "다만, 예외"])
    assert first.article == "제1조"
    assert first.paragraph is None and first.item is None
    assert first.document_version == doc.document_version
    assert first.source_uri == doc.source_uri
    assert first.printed_page_start is None


def test_untitled_articles_and_supplementary_namespace(tmp_path):
    texts = ["제1조 본문\n부칙 <시험 버전>\n제1조 부칙 내용\n"]
    path, doc = fixture(tmp_path, texts)
    chunks = prepare_pdf(path, doc, page_count=1, extractor=lambda _: texts)
    assert [c.is_supplementary for c in chunks] == [False, True]
    assert [c.article for c in chunks] == ["제1조", "제1조"]
    assert "부칙" not in chunks[0].content
    assert chunks[0].parent_article_id != chunks[1].parent_article_id
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_stable_ids_change_with_version_content_and_scope(tmp_path):
    texts = ["제12조의2(시험) 본문"]
    path, doc = fixture(tmp_path, texts)
    def run(d=doc, t=texts):
        return prepare_pdf(path, d, page_count=1, extractor=lambda _: t)[0]
    first = run()
    assert first == run()
    assert first.chunk_id != run(doc.model_copy(update={"document_version": "v2"})).chunk_id
    assert first.chunk_id != run(t=[texts[0] + " 수정"]).chunk_id
    assert first.chunk_id != run(doc.model_copy(update={"processed_scope": "partial"})).chunk_id


def test_explicit_page_header_and_division_are_excluded(tmp_path):
    texts = ["확인된 머리말\n제1장 시험\n제1조(시험) 본문\n", "확인된 머리말\n이어짐\n제2장 시험\n제2조(시험) 다음"]
    path, doc = fixture(tmp_path, texts)
    chunks = prepare_pdf(path, doc, page_count=2, extractor=lambda _: texts,
                         running_header="확인된 머리말")
    assert chunks[0].content == "제1조(시험) 본문\n이어짐"
    assert chunks[0].page_end == 2


def test_hash_is_verified_before_injected_extractor(tmp_path):
    path, doc = fixture(tmp_path, [])
    path.write_bytes(b"changed")
    def never(_):
        pytest.fail("extractor must not run on hash mismatch")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_pdf(path, doc, page_count=1, extractor=never)


def test_no_articles_and_bad_page_order_are_rejected(tmp_path):
    path, doc = fixture(tmp_path, [])
    with pytest.raises(ValueError, match="article"):
        prepare_pdf(path, doc, page_count=1, extractor=lambda _: ["plain unknown text"])
    source = SourceManifest(doc.document_id, doc.document_version, doc.source_uri, doc.pdf_sha256, 2)
    with pytest.raises(ValueError, match="pages"):
        chunk_document(ExtractedDocument(source, (PdfPage(2, "제1조 본문"), PdfPage(1, "내용"))), doc)
    with pytest.raises(ValueError, match="provenance"):
        chunk_document(ExtractedDocument(replace(source, document_version="other"),
                                         (PdfPage(1, "제1조 본문"), PdfPage(2, "내용"))), doc)


def test_pdf_cli_is_local_and_preserves_contract(tmp_path, capsys):
    path, doc = fixture(tmp_path, [])
    metadata = tmp_path / "document.json"
    metadata.write_text(doc.model_dump_json(), encoding="utf-8")
    assert main(["--pdf", str(path), "--metadata", str(metadata), "--page-count", "1"],
                extractor=lambda _: ["제1조(시험) ① 조건\n1. 항목\n다만, 예외"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dry-run"
    assert result["document"] == doc.model_dump()
    assert Chunk.model_validate(result["chunks"][0]).article == "제1조"


def test_pdf_cli_refuses_write_without_verified_external_settings(tmp_path, capsys):
    path, doc = fixture(tmp_path, [])
    metadata = tmp_path / "document.json"
    metadata.write_text(doc.model_dump_json(), encoding="utf-8")
    assert main(["--pdf", str(path), "--metadata", str(metadata), "--page-count", "1", "--write"],
                extractor=lambda _: pytest.fail("must stop before extraction")) == 1
    assert "verified" in capsys.readouterr().err


class FakeEmbedding:
    model = "synthetic-embedding"

    def __init__(self):
        self.inputs = []

    def embed_documents(self, texts):
        self.inputs = texts
        return [[1.0, 0.5] for _ in texts]


class FakeStore:
    def __init__(self, existing=False):
        self.existing = existing
        self.created = []
        self.points = []

    def collection_exists(self, **kwargs):
        return self.existing

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        return True

    def upsert(self, *, collection_name, points, wait):
        from types import SimpleNamespace
        self.points = points
        return SimpleNamespace(status="completed")

    def retrieve(self, **kwargs):
        return self.points


def test_pdf_pipeline_injects_embedding_and_store_with_full_provenance(tmp_path):
    texts = ["제1조(시험) ① 조건\n1. 항목\n다만, 예외"]
    path, doc = fixture(tmp_path, texts)
    model, store = FakeEmbedding(), FakeStore()
    result = ingest_pdf(path, doc, page_count=1, extractor=lambda _: texts,
                        write=True, collection="data_ingestion_fixture_v1",
                        provider="fake", expected_dimensions=2,
                        embedding_model=model, qdrant_client=store)
    assert result["verified_count"] == 1
    assert model.inputs == texts
    assert store.created[0]["vectors_config"].size == 2
    payload = store.points[0].payload
    contract = {k: payload[k] for k in Chunk.model_fields}
    chunk = Chunk.model_validate(contract)
    assert chunk.document_version == doc.document_version
    assert payload["pdf_sha256"] == doc.pdf_sha256
    assert payload["processed_scope"] == doc.processed_scope
    assert chunk.content == texts[0]


@pytest.mark.parametrize("existing,dimensions", [(True, 2), (False, 3)])
def test_pdf_pipeline_refuses_existing_collection_or_dimension_mismatch(tmp_path, existing, dimensions):
    path, doc = fixture(tmp_path, [])
    model, store = FakeEmbedding(), FakeStore(existing)
    with pytest.raises(ValueError, match="exists|dimensions"):
        ingest_pdf(path, doc, page_count=1, extractor=lambda _: ["제1조 시험"],
                   write=True, collection="data_ingestion_fixture_v1", provider="fake",
                   expected_dimensions=dimensions, embedding_model=model, qdrant_client=store)
    assert not store.created and not store.points
    if existing:
        assert not model.inputs


def test_pdf_dry_run_never_calls_embedding_or_store(tmp_path):
    path, doc = fixture(tmp_path, [])
    class Never:
        def __getattr__(self, name):
            pytest.fail("dry-run accessed an external dependency")
    result = ingest_pdf(path, doc.model_copy(update={"processed_scope": "partial"}),
                        page_count=1, extractor=lambda _: ["제1조 시험"],
                        embedding_model=Never(), qdrant_client=Never())
    assert result["mode"] == "dry-run"
    assert result["document"]["processed_scope"] == "partial"
    assert result["chunks"][0]["is_supplementary"] is None


def test_write_verification_rejects_corrupted_provenance(tmp_path):
    path, doc = fixture(tmp_path, [])
    class CorruptStore(FakeStore):
        def retrieve(self, **kwargs):
            from types import SimpleNamespace
            return [SimpleNamespace(id=p.id, payload={**p.payload, "page_start": 99})
                    for p in self.points]
    store = CorruptStore()
    with pytest.raises(ValueError, match="payloads do not match"):
        ingest_pdf(path, doc, page_count=1, extractor=lambda _: ["제1조 시험"],
                   write=True, collection="data_ingestion_fixture_v1", provider="fake",
                   expected_dimensions=2, embedding_model=FakeEmbedding(), qdrant_client=store)
