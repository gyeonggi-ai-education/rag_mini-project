"""Validate chunks locally by default; --write explicitly creates a new index."""

import argparse
import json
from pathlib import Path
import sys

# Allow the documented `python scripts/ingest.py` invocation from the root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.contracts import Document
from common.ingestion import EmbeddingError, IngestionError, ingest_chunks, ingest_pdf


def main(argv=None, *, embedding_model=None, qdrant_client=None, extractor=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=Path, nargs="?", help="Previous step's chunk JSONL")
    parser.add_argument("--pdf", type=Path, help="Extract a verified PDF (default: local dry-run)")
    parser.add_argument("--metadata", type=Path, help="JSON matching the shared Document contract")
    parser.add_argument("--page-count", type=int, help="Confirmed physical PDF page count")
    parser.add_argument("--running-header", help="Explicitly confirmed header to exclude")
    parser.add_argument("--verified-dimensions", type=int, help="Externally verified embedding dimension; required for PDF writes")
    parser.add_argument("--write", action="store_true", help="Create and verify a new collection (default: dry-run)")
    parser.add_argument("--collection", help="Explicit data-only version name: data_ingestion_<version>")
    parser.add_argument("--provider", default="monorouter", help="Embedding provider label (not an endpoint)")
    args = parser.parse_args(argv)
    try:
        if args.pdf is not None:
            if args.write and (args.verified_dimensions is None or args.verified_dimensions < 1):
                raise IngestionError("PDF write requires verified positive embedding dimensions")
            if args.chunks is not None or args.metadata is None or args.page_count is None:
                raise IngestionError("PDF mode requires metadata and page count, without chunk JSONL")
            try:
                document = Document.model_validate_json(args.metadata.read_text(encoding="utf-8"))
            except Exception:
                raise IngestionError("Cannot read valid document metadata") from None
            result = ingest_pdf(
                args.pdf, document, page_count=args.page_count, extractor=extractor,
                running_header=args.running_header, write=args.write,
                collection=args.collection, provider=args.provider,
                expected_dimensions=args.verified_dimensions,
                embedding_model=embedding_model, qdrant_client=qdrant_client,
            )
        else:
            if args.chunks is None or any(v is not None for v in (
                args.metadata, args.page_count, args.running_header, args.verified_dimensions,
            )):
                raise IngestionError("Specify chunk JSONL or PDF mode")
            result = ingest_chunks(
                args.chunks, write=args.write, collection=args.collection,
                provider=args.provider, embedding_model=embedding_model, qdrant_client=qdrant_client,
            )
    except (IngestionError, EmbeddingError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception:
        # Never expose dependency errors, endpoints, credentials or tracebacks.
        print("Ingestion failed", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
