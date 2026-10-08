import json
import re

from common.ai_model import get_llm_model

NO_ANSWER = "제공된 법률 문서에서는 해당 내용을 확인하기 어렵습니다."

SYSTEM_PROMPT = """너는 AI 기본법 안내 도우미다. 아래 [근거]에 있는 내용만 사용해서 질문에 답한다.
- 근거에 없는 내용은 추측하지 않는다. 근거만으로 답할 수 없으면 can_answer를 false로 한다.
- 법을 잘 모르는 사용자가 이해할 수 있게 쉬운 말로 설명한다.
- 핵심만 2~3문장으로 짧게 요약한다. 항목이 많은 목록은 모두 나열하지 말고 대표 항목만 들거나 "세부 항목은 아래 근거 조문을 확인하세요"라고 안내한다.
- 요약하더라도 적용 범위를 바꾸는 조건(예: "대통령령으로 정하는" 위임, "각 목의 어느 하나", 대상·영역의 한정)은 빠뜨리지 말고 한 번은 언급한다. 한정된 범위를 "예를 들면"처럼 예시로 바꿔 말하지 않는다.
- answer는 마크다운 서식(**굵게**, #, 목록 기호 등) 없이 평문으로 쓴다.
- 답변에 실제로 사용한 근거의 번호만 used에 넣는다.
반드시 아래 JSON 형식으로만 출력한다.
{"can_answer": true 또는 false, "answer": "답변", "used": [근거 번호 목록]}"""

_llm = get_llm_model(max_tokens=1024)


def build_context(chunks: list[dict]) -> str:
    """검색 결과를 번호가 붙은 근거 텍스트로 만든다."""
    return "\n\n".join(
        f"[근거 {i}] {c.get('article', '')} {c.get('article_title') or c.get('title', '')}\n{c['content']}"
        for i, c in enumerate(chunks, start=1)
    )


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0))


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """근거 기반 답변을 만들고, 실제 사용한 근거만 sources·used_chunk_ids로 돌려준다."""
    if not chunks:
        return {"answer": NO_ANSWER, "sources": [], "is_answerable": False, "used_chunk_ids": []}

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"[근거]\n{build_context(chunks)}\n\n[질문]\n{question}"),
    ]
    parsed = _parse(_llm.invoke(messages).content)

    if not parsed.get("can_answer"):
        return {"answer": NO_ANSWER, "sources": [], "is_answerable": False, "used_chunk_ids": []}

    used = list(dict.fromkeys(i for i in parsed.get("used", []) if isinstance(i, int) and 1 <= i <= len(chunks)))
    sources = [
        {"article": chunks[i - 1].get("article", ""), "content": chunks[i - 1]["content"]}
        for i in used
    ]
    used_chunk_ids = [chunks[i - 1].get("chunk_id") for i in used]
    return {"answer": parsed["answer"], "sources": sources, "is_answerable": True, "used_chunk_ids": used_chunk_ids}
