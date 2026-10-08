"""평가 전 Qdrant 컬렉션 점검 (Qdrant만 조회, MonoRouter 호출 없음)

- 접속, 벡터 차원·거리, 청크 수
- C 평가에 필요한 payload 필드 (필수 chunk_id·article·content / 선택 article_title·paragraph·is_supplementary)
- chunk_id 중복, 골든셋 정답 조문이 컬렉션에 있는지
- (선택) A가 공유한 청크 JSONL과 payload가 같은지

실행:
    uv run python -m evaluation.check_collection
    uv run python -m evaluation.check_collection --collection data_ingestion_law21311_v1 \
        --chunks rag-team-share/data/ingestion/chunks.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.chunk_source import resolve_collection

REQUIRED = ("chunk_id", "article", "content")
OPTIONAL = ("article_title", "paragraph", "is_supplementary")
EXPECTED_DIMENSIONS = 1536  # text-embedding-3-small (질문 임베딩과 같아야 함)
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_set.jsonl"


def scroll_all(client, collection: str) -> list:
    points, offset = [], None
    while True:
        batch, offset = client.scroll(collection_name=collection, limit=256, offset=offset,
                                      with_payload=True, with_vectors=False)
        points += batch
        if offset is None:
            return points


def check(collection: str | None = None, golden: Path = DEFAULT_GOLDEN, chunks_path: Path | None = None) -> dict:
    from common.qdrant import get_qdrant_client

    collection = resolve_collection(collection)
    client = get_qdrant_client()
    problems, notes = [], []

    if not client.collection_exists(collection):
        return {"collection": collection, "ok": False, "problems": [f"컬렉션 없음: {collection}"], "notes": []}

    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    if vectors.size != EXPECTED_DIMENSIONS:
        problems.append(f"벡터 차원 {vectors.size} (질문 임베딩은 {EXPECTED_DIMENSIONS})")
    if str(vectors.distance).lower().find("cosine") < 0:
        notes.append(f"거리 방식 {vectors.distance} (C 평가는 코사인 기준으로 해석)")

    points = scroll_all(client, collection)
    payloads = [p.payload or {} for p in points]
    missing = {f: sum(1 for p in payloads if not p.get(f)) for f in REQUIRED}
    for field, count in missing.items():
        if count:
            problems.append(f"필수 필드 {field} 없음: {count}개 청크")
    optional = {f: sum(1 for p in payloads if p.get(f) is not None) for f in OPTIONAL}
    duplicates = [cid for cid, n in Counter(p.get("chunk_id") for p in payloads).items() if cid and n > 1]
    if duplicates:
        problems.append(f"chunk_id 중복 {len(duplicates)}개")

    articles = {p.get("article") for p in payloads}
    gold_articles = []
    if golden and Path(golden).exists():
        for line in open(golden, encoding="utf-8"):
            if line.strip():
                gold_articles += json.loads(line).get("answer_articles", [])
    missing_gold = sorted(set(gold_articles) - articles, key=lambda a: (len(a), a))
    if missing_gold:
        problems.append(f"골든셋 정답 조문이 컬렉션에 없음: {', '.join(missing_gold)}")

    lengths = sorted(((len(p.get("content", "")), p.get("article")) for p in payloads), reverse=True)
    if lengths and lengths[0][0] > 1500:
        notes.append("긴 청크 (1,500자 초과, 검색이 어려울 수 있음): "
                     + ", ".join(f"{a} {n}자" for n, a in lengths if n > 1500))

    if chunks_path:
        shared = {}
        for line in open(chunks_path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                shared[row["chunk_id"]] = row
        stored = {p.get("chunk_id"): p for p in payloads}
        if set(shared) != set(stored):
            problems.append(f"공유 JSONL과 chunk_id 목록이 다름 (JSONL {len(shared)}, 컬렉션 {len(stored)})")
        else:
            differ = [cid for cid in shared if any(shared[cid].get(k) != stored[cid].get(k) for k in shared[cid])]
            if differ:
                problems.append(f"공유 JSONL과 payload가 다른 청크 {len(differ)}개")
            else:
                notes.append(f"공유 JSONL과 payload 일치 ({len(shared)}개)")

    sample = payloads[0] if payloads else {}
    return {
        "collection": collection,
        "ok": not problems,
        "points": len(points),
        "dimensions": vectors.size,
        "distance": str(vectors.distance),
        "articles": len(articles),
        "optional_fields": optional,
        "data_version": {k: sample.get(k) for k in ("document_version", "chunking_version", "schema_version")
                         if k in sample},
        "golden_articles_covered": f"{len(set(gold_articles)) - len(missing_gold)}/{len(set(gold_articles))}",
        "problems": problems,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=None)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--chunks", type=Path, default=None, help="A가 공유한 chunks.jsonl (payload 대조)")
    args = parser.parse_args()
    result = check(args.collection, args.golden, args.chunks)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n점검 결과:", "통과" if result["ok"] else "문제 있음")


if __name__ == "__main__":
    main()
