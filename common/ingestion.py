"""PDF extraction and provenance for the internal data-ingestion MVP.

Document identifiers/version strings are supplied by the caller; this module
does not infer legal versions or define the shared team schema. Page text is
kept verbatim, including headers, footers and blank pages, for later chunking.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import subprocess
import re
from uuid import NAMESPACE_URL, uuid5

from common.contracts import Chunk, Document


class IngestionError(ValueError):
    """Local validation/verification failure without external error bodies."""


def load_chunks(path: str | Path) -> list[dict]:
    """Read the previous step's flat JSONL without altering provenance."""
    try:
        chunks = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    except (OSError, UnicodeError, ValueError):
        raise IngestionError("Cannot read chunk JSONL") from None
    required = (
        "schema_version", "chunk_id", "document_id", "document_version",
        "pdf_sha256", "source_uri", "content", "normalization_version", "chunking_version",
    )
    if not chunks:
        raise IngestionError("Expected nonempty chunk JSONL")
    ids = set()
    identities = set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or any(
            not isinstance(chunk.get(key), str) or not chunk[key].strip()
            for key in required
        ):
            raise IngestionError("Missing chunk identity, content or provenance")
        if not re.fullmatch(r"[0-9a-f]{64}", chunk["pdf_sha256"]):
            raise IngestionError("Expected lowercase PDF SHA-256")
        start, end = chunk.get("page_start"), chunk.get("page_end")
        if type(start) is not int or type(end) is not int or not 1 <= start <= end:
            raise IngestionError("Invalid physical page range")
        spans = chunk.get("source_spans")
        if not isinstance(spans, list) or not spans:
            raise IngestionError("Missing source spans")
        for span in spans:
            if (
                not isinstance(span, dict)
                or any(type(span.get(k)) is not int for k in ("page", "start", "end"))
                or not start <= span["page"] <= end
                or not 0 <= span["start"] < span["end"]
            ):
                raise IngestionError("Invalid source span")
        if spans[0]["page"] != start or spans[-1]["page"] != end:
            raise IngestionError("Source span page range mismatch")
        if chunk["chunk_id"] in ids:
            raise IngestionError("Duplicate chunk ID")
        ids.add(chunk["chunk_id"])
        identities.add(tuple(chunk[k] for k in (
            "document_id", "document_version", "pdf_sha256", "schema_version",
            "normalization_version", "chunking_version",
        )))
    if len(identities) != 1:
        raise IngestionError("Expected one document and chunking version")
    return chunks


def ingest_chunks(
    path: str | Path, *, write: bool = False, collection: str | None = None,
    provider: str = "monorouter", embedding_model=None, qdrant_client=None,
) -> dict:
    """Dry-run locally, or create a new data-only collection once and verify it.

    This does not resume, delete, recreate or alias collections. A failure after
    creation leaves the collection for manual review; reruns refuse to write it.
    The data_ingestion_ prefix is an MVP local convention, not a team reservation.
    """
    return _store_chunks(
        load_chunks(path), write=write, collection=collection, provider=provider,
        embedding_model=embedding_model, qdrant_client=qdrant_client,
    )


def _store_chunks(
    chunks: Sequence[Mapping], *, write: bool, collection: str | None,
    provider: str, embedding_model=None, qdrant_client=None,
    expected_dimensions: int | None = None,
) -> dict:
    """Shared non-destructive storage boundary for legacy and contract payloads."""
    if not write:
        return {"mode": "dry-run", "chunk_count": len(chunks),
                "document_id": chunks[0]["document_id"],
                "document_version": chunks[0]["document_version"]}
    if not isinstance(collection, str) or not re.fullmatch(
        r"data_ingestion_[A-Za-z0-9][A-Za-z0-9_-]*", collection
    ):
        raise IngestionError("Explicit new collection name must start with data_ingestion_")
    try:
        if qdrant_client is None:
            from common.qdrant import get_qdrant_client

            qdrant_client = get_qdrant_client()
        if qdrant_client.collection_exists(collection_name=collection):
            raise IngestionError("Target collection already exists; write refused")
    except IngestionError:
        raise
    except Exception:
        raise IngestionError("Qdrant availability check failed") from None
    embedded = embed_chunks(chunks, provider=provider, embedding_model=embedding_model)
    if expected_dimensions is not None and embedded.record["dimensions"] != expected_dimensions:
        raise IngestionError("Observed embedding dimensions differ from verified dimensions")
    from qdrant_client.models import Distance, PointStruct, VectorParams

    points = [
        PointStruct(
            id=str(uuid5(NAMESPACE_URL, "rag-minipjt:data-ingestion:" + chunk["chunk_id"])),
            vector=list(vector), payload=dict(chunk),
        )
        for chunk, vector in zip(embedded.chunks, embedded.vectors, strict=True)
    ]
    try:
        created = qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=embedded.record["dimensions"], distance=Distance.COSINE),
        )
        if not created:
            raise IngestionError("Collection creation was not confirmed")
        result = qdrant_client.upsert(collection_name=collection, points=points, wait=True)
        if result.status != "completed":
            raise IngestionError("Upsert completion was not confirmed")
        records = qdrant_client.retrieve(
            collection_name=collection, ids=[p.id for p in points],
            with_payload=True, with_vectors=False,
        )
        expected = {p.id: p.payload for p in points}
        actual = {str(record.id): record.payload for record in records}
        if len(records) != len(points) or actual != expected:
            raise IngestionError("Stored IDs or source payloads do not match")
    except IngestionError:
        raise
    except Exception:
        raise IngestionError("Qdrant ingestion or verification failed") from None
    return {"mode": "write", "collection": collection,
            **embedded.record, "verified_count": len(points)}


class EmbeddingError(ValueError):
    """Embedding failed; messages exclude provider responses and secrets."""


@dataclass(frozen=True, slots=True)
class EmbeddedChunks:
    chunks: tuple[Mapping, ...]
    vectors: tuple[tuple[float, ...], ...]
    record: dict


def embed_chunks(
    chunks: Sequence[Mapping], *, provider: str, embedding_model=None,
    model: str | None = None,
) -> EmbeddedChunks:
    """Embed content in input order using the shared factory or an injected model.

    Provider is supplied by the caller, never inferred from a secret base URL.
    Model defaults to the adapter's configured model; explicit model labels are
    available for injected adapters without that attribute. Dimensions are
    observed from the response, not assumed from a model name. No DB is called.
    """
    chunks = tuple(chunks)
    if not chunks or any(
        not isinstance(chunk, Mapping)
        or not isinstance(chunk.get("content"), str)
        or not chunk["content"].strip()
        for chunk in chunks
    ):
        raise EmbeddingError("Expected nonempty chunks with content")
    if not isinstance(provider, str) or not provider.strip():
        raise EmbeddingError("Expected provider label")
    try:
        if embedding_model is None:
            from common.ai_model import get_embedding_model

            embedding_model = get_embedding_model()
    except Exception:
        raise EmbeddingError("Embedding model initialization failed") from None
    model_name = model if model is not None else getattr(embedding_model, "model", None)
    if not isinstance(model_name, str) or not model_name.strip():
        raise EmbeddingError("Expected model label")
    try:
        vectors = embedding_model.embed_documents([chunk["content"] for chunk in chunks])
    except Exception:
        raise EmbeddingError("Embedding request failed") from None
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
        raise EmbeddingError("Expected vector sequence")
    if len(vectors) != len(chunks):
        raise EmbeddingError("Vector count mismatch")
    validated = []
    dimensions = None
    for vector in vectors:
        if (
            not isinstance(vector, Sequence)
            or isinstance(vector, (str, bytes))
            or not vector
        ):
            raise EmbeddingError("Expected nonempty vector")
        if dimensions is None:
            dimensions = len(vector)
        if len(vector) != dimensions:
            raise EmbeddingError("Vector dimensions mismatch")
        converted = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise EmbeddingError("Expected finite numeric vector values")
            try:
                number = float(value)
            except (ValueError, OverflowError):
                raise EmbeddingError("Expected finite numeric vector values") from None
            if not math.isfinite(number):
                raise EmbeddingError("Expected finite numeric vector values")
            converted.append(number)
        validated.append(tuple(converted))
    return EmbeddedChunks(
        chunks=chunks,
        vectors=tuple(validated),
        record={
            "provider": provider, "model": model_name,
            "dimensions": dimensions, "chunk_count": len(chunks),
        },
    )


class PdfExtractionError(ValueError):
    """Extraction failed; messages omit external error bodies and file paths."""


@dataclass(frozen=True, slots=True)
class SourceManifest:
    document_id: str
    document_version: str
    source_uri: str
    pdf_sha256: str
    page_count: int

    def __post_init__(self):
        for value in (self.document_id, self.document_version, self.source_uri):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Source identifiers and version must be nonempty")
        if (
            not isinstance(self.pdf_sha256, str)
            or len(self.pdf_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.pdf_sha256)
        ):
            raise ValueError("Expected lowercase SHA-256 digest")
        if type(self.page_count) is not int or self.page_count < 1:
            raise ValueError("Expected positive physical page count")


@dataclass(frozen=True, slots=True)
class PdfPage:
    page: int  # Physical PDF page, 1-based; not an inferred printed page.
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    source: SourceManifest
    pages: tuple[PdfPage, ...]


def _poppler_pages(path: Path) -> list[str]:
    """Poppler separates physical pages with form feeds, including blank pages."""
    result = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
        capture_output=True,
        check=True,
        timeout=60,
    )
    pages = result.stdout.decode("utf-8").split("\f")
    if pages[-1] == "":
        pages.pop()  # Terminal separator, not a physical page.
    return pages


def extract_pdf(
    path: str | Path,
    manifest: SourceManifest,
    *,
    extractor: Callable[[Path], Sequence[str]] | None = None,
) -> ExtractedDocument:
    """Verify the source before extraction and return all physical page texts.

    An injected extractor returns one string per physical page in file order.
    Otherwise, installed Poppler is used. No model or database is called.
    """
    path = Path(path)
    try:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        raise PdfExtractionError("Cannot read source PDF") from None
    if digest != manifest.pdf_sha256:
        raise PdfExtractionError("SHA-256 mismatch")

    try:
        texts = (extractor or _poppler_pages)(path)
        if isinstance(texts, (str, bytes)):
            raise TypeError("Expected page sequence")
        texts = tuple(texts)
    except Exception:
        raise PdfExtractionError("PDF text extraction failed") from None

    if len(texts) != manifest.page_count:
        raise PdfExtractionError("Page count mismatch")
    if any(not isinstance(text, str) for text in texts):
        raise PdfExtractionError("Expected text for each physical page")
    return ExtractedDocument(
        source=manifest,
        pages=tuple(PdfPage(page, text) for page, text in enumerate(texts, start=1)),
    )


def prepare_pdf(
    path: str | Path, document: Document, *, page_count: int,
    extractor: Callable[[Path], Sequence[str]] | None = None,
    running_header: str | None = None,
) -> list[Chunk]:
    """Verify an explicitly identified PDF and produce the shared Chunk contract.

    No provider or database is initialized. Titles and versions come from the
    caller, not from filenames. Partial-document scope remains in the Document.
    """
    manifest = SourceManifest(
        document.document_id, document.document_version, document.source_uri,
        document.pdf_sha256.lower(), page_count,
    )
    return chunk_document(extract_pdf(path, manifest, extractor=extractor), document,
                          running_header=running_header)


def ingest_pdf(
    path: str | Path, document: Document, *, page_count: int,
    extractor: Callable[[Path], Sequence[str]] | None = None,
    running_header: str | None = None, write: bool = False,
    collection: str | None = None, provider: str = "monorouter",
    expected_dimensions: int | None = None, embedding_model=None, qdrant_client=None,
) -> dict:
    """Injectable PDF-to-store pipeline; default is entirely local.

    Writes require caller-confirmed external dimensions and an explicit new
    version collection. Observed dimensions must match before any DB mutation.
    This requirement does not itself establish provider support or reservation.
    """
    if write and (type(expected_dimensions) is not int or expected_dimensions < 1):
        raise IngestionError("PDF write requires verified positive embedding dimensions")
    chunks = prepare_pdf(path, document, page_count=page_count, extractor=extractor,
                         running_header=running_header)
    payloads = [dict(c.model_dump(), pdf_sha256=document.pdf_sha256,
                     processed_scope=document.processed_scope, title=document.title)
                for c in chunks]
    result = _store_chunks(
        payloads, write=write, collection=collection, provider=provider,
        expected_dimensions=expected_dimensions, embedding_model=embedding_model,
        qdrant_client=qdrant_client,
    )
    result["document"] = document.model_dump()
    if not write:
        result["chunks"] = [c.model_dump() for c in chunks]
    return result


def chunk_document(
    extracted: ExtractedDocument, document: Document, *,
    running_header: str | None = None,
) -> list[Chunk]:
    """Keep whole articles: paragraphs, items, conditions and exceptions stay together.

    Multiple paragraph/item numbers remain verbatim in content; their singular
    metadata fields stay None rather than attributing a whole article to one.
    Only line-start article markers are recognized, including untitled ones.
    No printed pages, missing structure, or supplementary article are inferred.
    """
    source = extracted.source
    if (source.document_id, source.document_version, source.source_uri, source.pdf_sha256) != (
        document.document_id, document.document_version, document.source_uri,
        document.pdf_sha256.lower(),
    ):
        raise IngestionError("Extracted provenance differs from document")
    if tuple(p.page for p in extracted.pages) != tuple(range(1, source.page_count + 1)):
        raise IngestionError("Expected all physical pages in order")
    article_pattern = re.compile(r"^[ \t]*(제\d+조(?:의\d+)?)(?=\(|\s|$)")
    division_pattern = re.compile(r"^[ \t]*제\d+(?:장|절)(?=\s|$)")
    supplement_pattern = re.compile(r"^[ \t]*부[ \t]*칙(?=\s|<|\(|$)")
    article = None
    supplement = 0
    occurrence = 0
    spans: list[dict] = []
    chunks: list[Chunk] = []

    def finish():
        nonlocal spans
        if article is None:
            return
        parts = []
        used_spans = []
        for span in spans:
            text = extracted.pages[span["page"] - 1].text
            start, end = span["start"], span["end"]
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                parts.append(text[start:end])
                used_spans.append({"page": span["page"], "start": start, "end": end})
        if not parts:
            raise IngestionError("Empty article content")
        parent_key = json.dumps(
            [document.document_id, document.document_version, source.pdf_sha256,
             supplement, article, occurrence], ensure_ascii=False,
        )
        payload = dict(
            document_id=document.document_id, document_version=document.document_version,
            source_uri=document.source_uri, article=article,
            page_start=used_spans[0]["page"], page_end=used_spans[-1]["page"],
            parent_article_id=hashlib.sha256(parent_key.encode("utf-8")).hexdigest(),
            is_supplementary=(True if supplement else
                              False if document.processed_scope == "full" else None),
            content="\n".join(parts),
        )
        identity = json.dumps(
            {"document": document.model_dump(), "chunk": payload,
             "source_spans": used_spans, "chunking_version": "contract-whole-article-v1"},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        chunks.append(Chunk(chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                            **payload))
        spans = []

    for page in extracted.pages:
        offset = 0
        for line in page.text.splitlines(keepends=True):
            end = offset + len(line)
            # Only remove explicitly supplied furniture. Never guess what a
            # header/footer means in an unknown PDF.
            if running_header is not None and line.strip() == running_header:
                offset = end
                continue
            if supplement_pattern.match(line):
                finish()
                article = None
                supplement += 1
            elif division_pattern.match(line):
                finish()
                article = None
            else:
                heading = article_pattern.match(line)
                if heading:
                    finish()
                    article = heading.group(1)
                    occurrence += 1
                if article is not None:
                    if spans and spans[-1]["page"] == page.page and spans[-1]["end"] == offset:
                        spans[-1]["end"] = end
                    else:
                        spans.append({"page": page.page, "start": offset, "end": end})
            offset = end
    finish()
    if not chunks:
        raise IngestionError("No recognizable article structure; manual review required")
    return chunks
