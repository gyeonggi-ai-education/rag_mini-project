"""공통 데이터 계약 (project-plan.md 4장 기준 초안)

킥오프에서 합의 후 변경할 때는 팀에 먼저 공유한다.
"""

from typing import NotRequired, TypedDict


class Chunk(TypedDict):
    # 필수 필드
    chunk_id: str  # 문서 버전·조문 경로·분할 순서로 재현 가능하게 생성 (A 관리)
    document_id: str
    document_version: str  # 본문에서 확인한 법률 버전 표기, 확인 전에는 "미확정"
    article: str  # 원문 조문 표기 (예: "제2조")
    content: str  # 출처로 제시할 수 있는 실제 법률 본문

    # 선택 필드: 추출 가능한 값만 채우고, 구조가 없으면 None (추측하지 않음)
    law_name: NotRequired[str | None]
    article_title: NotRequired[str | None]
    paragraph: NotRequired[int | None]
    chapter: NotRequired[str | None]
    section: NotRequired[str | None]
    page_start: NotRequired[int | None]  # PDF 물리 페이지, 1부터 시작
    page_end: NotRequired[int | None]
    is_supplementary: NotRequired[bool | None]  # 부칙 여부


class RetrievedChunk(Chunk):
    """검색 결과: Chunk + 단계별 점수 (점수 의미가 다른 단계끼리 합산하지 않는다)"""

    retrieval_score: float  # dense면 유사도, multi_query면 RRF 점수
    rerank_score: float | None  # Rerank를 거치지 않았으면 None
    rank: int  # 1부터 시작하는 최종 순위
