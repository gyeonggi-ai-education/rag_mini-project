"""C 개인 테스트용 임시 적재 (A의 정식 청킹·적재 전까지만 사용)

법령 PDF를 단순 규칙(조 단위)으로 잘라 C 전용 임시 컬렉션에 적재한다.
A가 정식 Chunk JSONL을 주면 --jsonl 로 그 파일을 그대로 적재할 수 있다.

실행:
    uv run --with pypdf python -m evaluation.temp_ingest --pdf <법령.pdf>
    uv run python -m evaluation.temp_ingest --jsonl <chunks.jsonl>
"""

import argparse
import json
import re
import uuid
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from common.ai_model import get_embedding_model
from common.config import EMBEDDING_MODEL
from common.qdrant import get_qdrant_client

# 공용 law_articles를 덮어쓰지 않도록 C 전용 임시 컬렉션에 적재한다.
TEMP_COLLECTION = "tmp_c_law_articles"

DOCUMENT_ID = "ai_basic_law"
DATA_VERSION = "tmp-c-article-v0"

ARTICLE_RE = re.compile(r"^\s*(제\d+조(?:의\d+)?)\s*\(([^)]+)\)\s*", re.M)
SUPPLEMENTARY_RE = re.compile(r"^\s*부\s*칙", re.M)
# law.go.kr PDF의 머리말·쪽 번호 (예: "법제처 3 국가법령정보센터")
NOISE_RE = re.compile(r"^.*(법제처|국가법령정보센터).*$", re.M)


def chunks_from_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    text = NOISE_RE.sub("", text)

    supplementary = SUPPLEMENTARY_RE.search(text)
    supplementary_start = supplementary.start() if supplementary else len(text)

    matches = list(ARTICLE_RE.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        is_supplementary = m.start() >= supplementary_start
        if not is_supplementary:
            end = min(end, supplementary_start)
        article, title = m.group(1), m.group(2)
        prefix = "부칙" if is_supplementary else "본문"
        chunks.append({
            "chunk_id": f"{DATA_VERSION}:{prefix}:{article}",
            "document_id": DOCUMENT_ID,
            "document_version": "미확정",
            "article": article,
            "article_title": title,
            "paragraph": None,
            "is_supplementary": is_supplementary,
            "content": text[m.end():end].strip(),
        })
    return chunks


def chunks_from_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embedding_text(chunk: dict) -> str:
    title = chunk.get("article_title") or ""
    return f"{chunk['article']}({title}) {chunk['content']}"


def ingest(chunks: list[dict]):
    embeddings = get_embedding_model()
    client = get_qdrant_client()
    vectors = embeddings.embed_documents([embedding_text(c) for c in chunks])

    # 임시 컬렉션이므로 매번 새로 만든다 (공용 컬렉션은 건드리지 않음)
    if client.collection_exists(TEMP_COLLECTION):
        client.delete_collection(TEMP_COLLECTION)
    client.create_collection(
        collection_name=TEMP_COLLECTION,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        metadata={"embedding_model": EMBEDDING_MODEL, "data_version": DATA_VERSION},
    )
    client.upsert(
        collection_name=TEMP_COLLECTION,
        points=[
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])), vector=v, payload=c)
            for c, v in zip(chunks, vectors)
        ],
    )
    print(f"'{TEMP_COLLECTION}'에 {len(chunks)}개 청크 적재 (차원 {len(vectors[0])})")


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path)
    source.add_argument("--jsonl", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="청킹 결과만 확인하고 적재하지 않음")
    args = parser.parse_args()

    chunks = chunks_from_pdf(args.pdf) if args.pdf else chunks_from_jsonl(args.jsonl)
    body = [c for c in chunks if not c.get("is_supplementary")]
    print(f"청크 {len(chunks)}개 (본문 {len(body)}개, 부칙 {len(chunks) - len(body)}개)")
    for c in chunks[:3] + chunks[-2:]:
        print(f"  {c['chunk_id']} {c.get('article_title')} | {c['content'][:60]!r}")

    if not args.dry_run:
        ingest(chunks)


if __name__ == "__main__":
    main()
