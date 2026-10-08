"""C 평가용 임시 답변 생성 (B의 답변·출처 생성 함수 완성 전까지만 사용)

노트북 7번 셀(RagAnswer + structured output)을 기반으로, 답변이 실제로 사용한 청크를
청크 번호로 받아 chunk_id로 되돌린다.

B와 맞출 인터페이스:
    generate_answer(question, chunks) -> {"answer": str, "is_answerable": bool, "used_chunk_ids": list[str]}
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RagAnswer(BaseModel):
    answer: str = Field(description="사용자 질문에 대한 답변")
    used_chunk_numbers: list[int] = Field(description="답변의 근거로 실제 사용한 청크 번호 목록 (예: [1, 3])")
    is_answerable: bool = Field(description="주어진 컨텍스트만으로 답변 가능했는지 여부")


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """당신은 주어진 법률 조문만 근거로 답변하는 법률 QA 어시스턴트다.
        컨텍스트에 없는 내용은 추측하거나 지어내지 않는다.
        컨텍스트만으로 답변할 수 없으면 is_answerable을 false로 설정하고, 문서에서 확인하기 어렵다고 답한다.
        used_chunk_numbers에는 답변에 실제로 사용한 청크 번호만 넣는다."""
    ),
    ("human", "컨텍스트:\n{context}\n\n질문: {question}"),
])

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        from common.ai_model import get_llm_model

        _chain = PROMPT | get_llm_model().with_structured_output(RagAnswer, method="function_calling")
    return _chain


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[청크 {i}] {c['article']}({c.get('article_title') or ''})\n{c['content']}"
        for i, c in enumerate(chunks, start=1)
    )


def generate_answer(question: str, chunks: list[dict]) -> dict:
    result = _get_chain().invoke({"context": build_context(chunks), "question": question})
    used = [chunks[n - 1]["chunk_id"] for n in result.used_chunk_numbers if 1 <= n <= len(chunks)]
    return {
        "answer": result.answer,
        "is_answerable": result.is_answerable,
        "used_chunk_ids": list(dict.fromkeys(used)),  # 순서 유지 중복 제거
    }
