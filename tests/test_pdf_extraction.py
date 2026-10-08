"""PDF provenance tests use an injected extractor without external services."""

import hashlib

import pytest

from common.ingestion import PdfExtractionError, SourceManifest, extract_pdf


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"fake PDF bytes for injected extraction")
    manifest = SourceManifest(
        document_id="fixture-law",
        document_version="fixture-version",
        source_uri=path.as_uri(),
        pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        page_count=3,
    )
    return path, manifest


def test_extract_preserves_text_version_hash_and_physical_pages(source):
    path, manifest = source
    texts = ["제1조(목적)\n  실제 발췌\n", "", "제2조(정의)\n① 조건과 예외\n"]
    calls = []

    def fake_extractor(pdf_path):
        calls.append(pdf_path)
        return texts

    document = extract_pdf(path, manifest, extractor=fake_extractor)

    assert calls == [path]
    assert document.source == manifest
    assert document.source.document_version == "fixture-version"
    assert document.source.pdf_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [page.page for page in document.pages] == [1, 2, 3]
    assert [page.text for page in document.pages] == texts
    assert path.read_bytes() == b"fake PDF bytes for injected extraction"


def test_hash_mismatch_rejected_before_extraction(source):
    path, manifest = source
    path.write_bytes(b"changed source")

    def should_not_run(_):
        pytest.fail("Extractor must not run on a different PDF")

    with pytest.raises(PdfExtractionError, match="SHA-256 mismatch"):
        extract_pdf(path, manifest, extractor=should_not_run)


def test_page_count_mismatch_is_rejected(source):
    path, manifest = source
    with pytest.raises(PdfExtractionError, match="Page count mismatch"):
        extract_pdf(path, manifest, extractor=lambda _: ["only one page"])
