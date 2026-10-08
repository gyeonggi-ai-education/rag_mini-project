from qdrant_client import models

from common.ai_model import get_embedding_model
from common.config import CANDIDATE_K, COLLECTION_NAME
from common.qdrant import get_qdrant_client

_embedding = get_embedding_model()
_qdrant = get_qdrant_client()


def _to_chunks(hits) -> list[dict]:
    return [
        {
            **hit.payload,
            "point_id": hit.id,
            "retrieval_score": hit.score,
            "rerank_score": None,
            "rank": rank,
        }
        for rank, hit in enumerate(hits, start=1)
    ]


def embed_queries(texts: list[str]) -> list[list[float]]:
    """여러 질의를 한 번의 임베딩 요청으로 벡터화한다. 입력 순서를 유지한다."""
    return _embedding.embed_documents(texts)


def dense_retrieve(question: str, candidate_k: int = CANDIDATE_K, collection: str = COLLECTION_NAME) -> list[dict]:
    """질문을 임베딩해 Qdrant에서 유사한 Chunk를 찾는다. 점수는 retrieval_score에 담는다."""
    query_vector = _embedding.embed_query(question)
    hits = _qdrant.query_points(
        collection_name=collection,
        query=query_vector,
        limit=candidate_k,
        with_payload=True,
    ).points
    return _to_chunks(hits)


def dense_retrieve_many(
    questions: list[str], candidate_k: int = CANDIDATE_K, collection: str = COLLECTION_NAME
) -> list[list[dict]]:
    """질의 여러 개를 임베딩 1회·Qdrant 1회 요청으로 검색해 질의별 결과 목록을 돌려준다(Multi Query용)."""
    vectors = embed_queries(questions)
    responses = _qdrant.query_batch_points(
        collection_name=collection,
        requests=[models.QueryRequest(query=v, limit=candidate_k, with_payload=True) for v in vectors],
    )
    return [_to_chunks(r.points) for r in responses]
