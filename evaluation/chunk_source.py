"""chunk_id로 원본 청크(원문 전체)를 불러온다

평가 결과 파일에는 본문이 80자만 저장되므로, 원문은 적재된 Qdrant 컬렉션이나 청크 JSONL에서 가져온다.
Qdrant 클라이언트는 함수 호출 시점에만 불러온다 (import만으로 외부 연결이 생기지 않도록).

    from evaluation.chunk_source import get_chunks
    get_chunks(["<chunk_id>"])  # 기본: QDRANT_COLLECTION, 없으면 A의 data_ingestion_law21311_v1
    get_chunks(["<chunk_id>"], jsonl_path="rag-team-share/data/ingestion/chunks.jsonl")

터미널:
    uv run python -m evaluation.chunk_source d34421c4c405dba812a11ed6daaed3d442799b357fa2c4f44262c7f89244cc1d
"""

import argparse
import json
import os
from pathlib import Path

# A가 공유한 실제 데이터 컬렉션 (rag-team-share/data/qdrant/manifest.json).
# B와 같은 환경변수 QDRANT_COLLECTION 으로 바꿀 수 있다.
A_COLLECTION = "data_ingestion_law21311_v1"


def default_collection() -> str:
    import common.config  # noqa: F401  .env 로딩 (QDRANT_COLLECTION)

    return os.getenv("QDRANT_COLLECTION") or A_COLLECTION


def resolve_collection(collection: str | None) -> str:
    """None·"default" → QDRANT_COLLECTION 환경변수, 없으면 A의 컬렉션"""
    return default_collection() if collection in (None, "default") else collection


def _from_jsonl(chunk_ids: list[str], path: str | Path) -> dict[str, dict]:
    wanted = set(chunk_ids)
    found = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                if chunk.get("chunk_id") in wanted:
                    found[chunk["chunk_id"]] = chunk
    return found


def _from_qdrant(chunk_ids: list[str], collection: str) -> dict[str, dict]:
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    from common.qdrant import get_qdrant_client

    points, _ = get_qdrant_client().scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[FieldCondition(key="chunk_id", match=MatchAny(any=list(chunk_ids)))]),
        limit=len(chunk_ids),
        with_payload=True,
        with_vectors=False,
    )
    return {p.payload["chunk_id"]: {**p.payload, "point_id": p.id} for p in points}


def get_chunks(
    chunk_ids: list[str],
    collection: str | None = None,
    jsonl_path: str | Path | None = None,
) -> list[dict]:
    """요청 순서대로 원본 청크를 반환. 찾지 못한 청크는 found=False 로 표시한다."""
    chunk_ids = list(dict.fromkeys(chunk_ids))  # 순서 유지 중복 제거
    if not chunk_ids:
        return []
    if jsonl_path:
        found = _from_jsonl(chunk_ids, jsonl_path)
    else:
        found = _from_qdrant(chunk_ids, resolve_collection(collection))
    return [
        {**found[c], "found": True} if c in found else {"chunk_id": c, "found": False}
        for c in chunk_ids
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_ids", nargs="+")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--jsonl", default=None)
    args = parser.parse_args()

    for chunk in get_chunks(args.chunk_ids, args.collection, args.jsonl):
        if not chunk["found"]:
            print(f"[없음] {chunk['chunk_id']}")
            continue
        title = chunk.get("article_title") or ""
        print(f"[{chunk['chunk_id']}] {chunk['article']}({title})")
        print(chunk["content"])
        print()


if __name__ == "__main__":
    main()
