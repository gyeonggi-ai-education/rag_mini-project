"""Synthetic article fixtures; these are not legal evidence."""

from dataclasses import replace

from common.chunking import chunk_articles
from common.ingestion import ExtractedDocument, PdfPage, SourceManifest


def document(*texts):
    return ExtractedDocument(
        SourceManifest("synthetic-law", "v1", "fixture://synthetic", "a" * 64, len(texts)),
        tuple(PdfPage(i, text) for i, text in enumerate(texts, 1)),
    )


def reconstruct(doc, chunk):
    return "\n".join(
        doc.pages[span["page"] - 1].text[span["start"]:span["end"]]
        for span in chunk["source_spans"]
    )


def test_article_headings_and_inline_references():
    doc = document(
        "문서 안내\n제1장 총칙\n제1조(목적) 합성 내용\n"
        "제2조에 따른 조건은 유지한다.\n\n"
        "제17조의2(지원) ① 합성 조건\n② 합성 예외\n"
    )
    chunks = chunk_articles(doc)
    assert [c["article"] for c in chunks] == ["제1조", "제17조의2"]
    assert chunks[1]["article_title"] == "지원"
    assert "제2조에 따른 조건" in chunks[0]["content"]
    assert "② 합성 예외" in chunks[1]["content"]
    assert all(reconstruct(doc, c) == c["content"] for c in chunks)
    assert all(c["document_id"] == doc.source.document_id for c in chunks)


def test_multi_page_article_preserves_spans_and_skips_page_furniture():
    doc = document(
        "합성 문서\n제22조의2(연구소) ① 조건\n법제처   1   국가법령정보센터\n",
        "합성 문서\n② 이어지는 예외\n\n제23조(다음) 본문\n",
    )
    chunks = chunk_articles(doc, running_header="합성 문서")
    first = chunks[0]
    assert (first["page_start"], first["page_end"]) == (1, 2)
    assert [s["page"] for s in first["source_spans"]] == [1, 2]
    assert first["content"] == "제22조의2(연구소) ① 조건\n② 이어지는 예외"
    assert reconstruct(doc, first) == first["content"]
    assert "제23조" not in first["content"]


def test_ids_are_deterministic_and_change_with_version_or_content():
    doc = document("제1조(목적) 합성 본문\n")
    chunk = chunk_articles(doc)[0]
    assert chunk_articles(doc)[0]["chunk_id"] == chunk["chunk_id"]
    changed_version = replace(doc, source=replace(doc.source, document_version="v2"))
    changed_content = replace(doc, pages=(PdfPage(1, "제1조(목적) 다른 본문\n"),))
    assert chunk_articles(changed_version)[0]["chunk_id"] != chunk["chunk_id"]
    assert chunk_articles(changed_content)[0]["chunk_id"] != chunk["chunk_id"]
