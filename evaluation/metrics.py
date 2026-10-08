"""검색 지표 (project-plan.md 7장)

정답 근거는 조문(·항) 단위로 관리한다. 청킹 버전이 바뀌어도 평가 기준이 깨지지 않도록
검색 결과의 article/paragraph와 비교한다.
- paragraph가 None인 정답: 같은 조문의 어떤 청크든 정답으로 인정
- paragraph가 있는 정답: 조문 단위 청크(paragraph None)나 같은 항 청크를 정답으로 인정
- chunk_id가 있는 정답: 해당 청크만 정답으로 인정 (청킹 버전이 고정된 더미 평가용)
"""


def _matches(chunk: dict, evidence: dict) -> bool:
    if evidence.get("chunk_id"):
        return chunk["chunk_id"] == evidence["chunk_id"]
    if chunk["article"] != evidence["article"]:
        return False
    if bool(chunk.get("is_supplementary")) != bool(evidence.get("is_supplementary")):
        return False
    gold_paragraph = evidence.get("paragraph")
    chunk_paragraph = chunk.get("paragraph")
    return gold_paragraph is None or chunk_paragraph is None or chunk_paragraph == gold_paragraph


def is_relevant(chunk: dict, evidences: list[dict]) -> bool:
    """검색된 청크가 정답 근거 중 하나와 일치하는지"""
    return any(_matches(chunk, e) for e in evidences)


def found_evidence(results: list[dict], evidences: list[dict], k: int) -> list[int | None]:
    """정답 근거별로 처음 검색된 순위(1부터)를 반환, top-k 안에 없으면 None"""
    top = results[:k]
    ranks = []
    for evidence in evidences:
        rank = next((i for i, chunk in enumerate(top, start=1) if _matches(chunk, evidence)), None)
        ranks.append(rank)
    return ranks


def hit_at_k(results, evidences, k) -> float:
    return float(any(r is not None for r in found_evidence(results, evidences, k)))


def recall_at_k(results, evidences, k) -> float:
    ranks = found_evidence(results, evidences, k)
    return sum(r is not None for r in ranks) / len(ranks)


def reciprocal_rank(results, evidences, k) -> float:
    ranks = [r for r in found_evidence(results, evidences, k) if r is not None]
    return 1 / min(ranks) if ranks else 0.0
