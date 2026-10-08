"""Whole-article chunks for the internal MVP, without model or DB calls.

Spans index the verbatim PdfPage.text in Unicode code points, end exclusive.
No normalization is performed. A confirmed running header can be supplied;
the narrowly recognized Poppler law footer and chapter headings are excluded.
This is not a general supplementary-provisions parser or shared team schema.
"""

import hashlib
import json
import re

from common.ingestion import ExtractedDocument


ARTICLE = re.compile(r"^[ \t]*(제\d+조(?:의\d+)?)\(([^)\n]+)\)")
DIVISION = re.compile(r"^[ \t]*제\d+(?:장|절)[ \t]+")
FOOTER = re.compile(r"^[ \t]*법제처[ \t]+\d+[ \t]+국가법령정보센터[ \t]*$")


def chunk_articles(
    document: ExtractedDocument, *, running_header: str | None = None
) -> list[dict]:
    """Keep each titled article intact, including continuations on later pages.

    Text before the first article and standalone chapter/section headings are
    not article content. Content is reconstructed by joining source spans with
    a newline. IDs use canonical SHA-256, including document version and body.
    """
    if tuple(p.page for p in document.pages) != tuple(
        range(1, document.source.page_count + 1)
    ):
        raise ValueError("Expected all physical pages in order")
    chunks = []
    article = title = None
    spans = []

    def finish():
        nonlocal spans
        if article is None:
            return
        trimmed = []
        parts = []
        for span in spans:
            text = document.pages[span["page"] - 1].text
            start, end = span["start"], span["end"]
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                trimmed.append({"page": span["page"], "start": start, "end": end})
                parts.append(text[start:end])
        content = "\n".join(parts)
        source = document.source
        chunk = {
            "schema_version": "data-mvp-v1",
            "document_id": source.document_id,
            "document_version": source.document_version,
            "pdf_sha256": source.pdf_sha256,
            "source_uri": source.source_uri,
            "article": article,
            "article_title": title,
            "content": content,
            "page_start": trimmed[0]["page"],
            "page_end": trimmed[-1]["page"],
            "source_spans": trimmed,
            "normalization_version": "verbatim-v1",
            "chunking_version": "whole-article-v1",
        }
        identity = json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        chunk["chunk_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        chunks.append(chunk)
        spans = []

    for page in document.pages:
        offset = 0
        for line in page.text.splitlines(keepends=True):
            end = offset + len(line)
            heading = ARTICLE.match(line)
            if heading:
                finish()
                article, title = heading.groups()
            if DIVISION.match(line):
                finish()
                article = title = None
            furniture = (
                running_header is not None and line.strip() == running_header
            ) or FOOTER.fullmatch(line.rstrip("\r\n"))
            if article is not None and not furniture:
                if spans and spans[-1]["page"] == page.page and spans[-1]["end"] == offset:
                    spans[-1]["end"] = end
                else:
                    spans.append({"page": page.page, "start": offset, "end": end})
            offset = end
    finish()
    return chunks
