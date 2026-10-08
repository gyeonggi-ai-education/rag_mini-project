# B 구현 정보 요청 (C 평가 연결용)

C(Multi Query·평가)가 B의 Rerank·답변·`/ask`를 평가에 미리 연결하려고 필요한 정보입니다.
각 항목에 **C가 예상한 값**을 적어 두었습니다. 같으면 `[x] 같음`에 체크만, 다르면 "실제" 칸에 적어 주세요.
모르거나 아직 안 정했으면 "미정"이라고 적어 주세요.

- 작성자: B
- 작성일:
- 코드 위치 (브랜치·커밋, 또는 "아직 안 올림"):

---

## 1. Rerank

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 파일 | `retrieval/rerank.py` | [ ] | |
| 함수 이름 | `rerank` | [ ] | |
| 입력 | `rerank(question: str, candidates: list[dict], final_k: int)` | [ ] | |
| 입력 후보 수 | 20개 | [ ] | |
| 반환 | 청크 dict 목록 (입력 필드 유지) | [ ] | |
| 반환에 추가하는 점수 필드 | `rerank_score` | [ ] | |
| 반환에 추가하는 순위 필드 | `rank` (1부터) | [ ] | |
| 기존 Dense 점수 유지 | `retrieval_score` 그대로 둠 | [ ] | |

**Rerank 방식** (하나 체크):
- [ ] LLM 1회로 후보 전체 채점
- [ ] LLM을 후보마다 호출 (후보 20개면 20회)
- [ ] 전용 Rerank API (MonoRouter)
- [ ] 로컬 모델 (모델 이름: ___________ , 예: `BAAI/bge-reranker-v2-m3`)
- [ ] 기타: ___________

| 항목 | 작성 |
|---|---|
| Rerank 1회당 MonoRouter 호출 수 | |
| 사용하는 모델 ID | |
| 추가로 설치하는 패키지 | |
| 호출 실패 시 동작 (예: Dense 순서 그대로 반환 / 예외) | |

> 왜 필요한가: Rerank 방식에 따라 평가 1회의 호출 수가 약 18회(로컬)에서 400회 이상(후보마다 LLM)까지 달라집니다.
> MonoRouter는 분당 30회 제한이고 키를 여러 명이 함께 쓰고 있습니다.

## 2. 답변 생성

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 파일 | `generation/answer.py` | [ ] | |
| 함수 이름 | `generate_answer` | [ ] | |
| 입력 | `generate_answer(question: str, chunks: list[dict])` | [ ] | |
| 반환 형식 | `{"answer": str, "is_answerable": bool, "used_chunk_ids": list[str]}` | [ ] | |
| 출처 | 답변에 **실제로 사용한** 청크만 (검색 결과 전체 아님) | [ ] | |
| 답변 불가 처리 | 근거가 없으면 `is_answerable=False` + "문서에서 확인하기 어렵다" | [ ] | |
| LLM 호출 수 | 답변 1개당 1회 | [ ] | |

출처를 어떻게 정하는지 (하나 체크):
- [ ] LLM이 사용한 청크 번호·ID를 구조화 출력으로 돌려줌
- [ ] LLM이 인용한 조문 번호(예: "제2조")로 청크를 찾음
- [ ] 검색된 청크 전체를 출처로 붙임
- [ ] 기타: ___________

## 3. `/ask` 검색 흐름

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 지원 모드 | baseline / rerank / multi_query / combined | [ ] | |
| **최종 서비스 모드** (사용자에게 기본으로 쓰는 모드) | 미정 | | |
| Multi Query 함수 | C의 `retrieval/multi_query.py` `multi_query_retrieve` 사용 | [ ] | |
| combined 순서 | 확장 질의 → 질의별 Dense → RRF 병합 → **병합 후 Rerank 1회** | [ ] | |
| 질의별 Dense 후보 수 | 20 | [ ] | |
| 최종 K (답변 컨텍스트 청크 수) | 5 | [ ] | |
| 확장 질의 임베딩 | 여러 질의를 한 번의 요청으로 묶음 | [ ] | |
| 모드 선택 방법 | 내부 설정 (요청 형식은 `{"question": ...}` 그대로) | [ ] | |

**공유 함수가 있는지**:
- [ ] 있음 → 파일·함수: ___________ (예: `retrieve(question, mode) -> list[dict]`)
- [ ] 없음 / 만들 수 있음
- [ ] 없음 / 만들 계획 없음

> 왜 필요한가: 평가의 combined와 `/ask`가 같은 흐름이어야 평가 수치가 실제 서비스를 대표합니다.
> 공유 함수가 있으면 평가가 그 함수를 그대로 호출하도록 C가 바꾸겠습니다.

## 4. Dense·Qdrant (현재 받은 코드 기준 확인)

| 항목 | C 예상 (현재 `retrieval/dense.py`) | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 함수 | `dense_retrieve(question, candidate_k=20, collection="law_articles")` | [ ] | |
| 반환 필드 | payload 전체 + `point_id`, `retrieval_score`, `rerank_score=None`, `rank` | [ ] | |
| 기본 컬렉션 | `law_articles` | [ ] | |
| 질의 여러 개 묶음 임베딩 함수 | 없음 (추가 요청: 예 `embed_queries(texts)`) | [ ] | |

## 5. 공통 파일·설정

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| LLM 함수 | `common/ai_model.py` `get_llm_model()` | [ ] | |
| 임베딩 함수 | `common/ai_model.py` `get_embedding_model()` | [ ] | |
| 환경변수 이름 | `LLM_API_KEY`, `LLM_BASE_URL` (계획서는 `MONOROUTER_*`) | [ ] | |
| LLM·임베딩 `max_retries` | 기본값 (429 시 2회 재시도) → 0~1로 낮출지 | [ ] | |
| Qdrant 주소 | `.env`의 `QDRANT_URL` (A 서버로 바꿀 예정) | [ ] | |
| Qdrant API 키 지원 | 없음 (A가 Cloud·보안 설정 시 `common/qdrant.py`에 추가 필요) | [ ] | |

## 6. FastAPI

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 앱 파일 | `app/main.py` | [ ] | |
| `/ask` 입력 | `{"question": "..."}` | [ ] | |
| `/ask` 출력 | `{"answer": "...", "sources": [{"article": "제○조", "content": "..."}]}` | [ ] | |
| 평가 결과 라우터 연결 | `from evaluation.api import router` + `app.include_router(router)` (`HANDOFF_B.md` 2장) | [ ] | |
| 평가 결과 화면 | 만들 예정 / 안 만듦 / 미정 | | |

`/ask` 응답 예시를 하나 붙여 주세요 (실제 또는 예상):

```json

```

## 7. 일정·기타

| 항목 | 작성 |
|---|---|
| Rerank 코드 공유 예정 시점 | |
| 답변·`/ask` 코드 공유 예정 시점 | |
| C에게 요청하는 것 | |
| 기타 메모 | |
