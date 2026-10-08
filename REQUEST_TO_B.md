# B 구현 정보 요청 (C 평가 연결용)

C(Multi Query·평가)가 B의 Rerank·답변·`/ask`를 평가에 미리 연결하려고 필요한 정보입니다.
각 항목에 **C가 예상한 값**을 적어 두었습니다. 같으면 `[x] 같음`에 체크만, 다르면 "실제" 칸에 적어 주세요.
모르거나 아직 안 정했으면 "미정"이라고 적어 주세요.

- 작성자: B
- 작성일: 2026-10-08
- 코드 위치 (브랜치·커밋, 또는 "아직 안 올림"): 아직 안 올림 (로컬 작업 중, `main` 브랜치 미커밋 상태)

---

## 1. Rerank

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 파일 | `retrieval/rerank.py` | [x] | |
| 함수 이름 | `rerank` | [x] | |
| 입력 | `rerank(question: str, candidates: list[dict], final_k: int)` | [x] | `final_k`는 기본값 5가 있음 |
| 입력 후보 수 | 20개 | [x] | `CANDIDATE_K = 20` (`common/config.py`) |
| 반환 | 청크 dict 목록 (입력 필드 유지) | [x] | 상위 `final_k`개만 반환 |
| 반환에 추가하는 점수 필드 | `rerank_score` | [x] | 0~10점. 점수가 없는 후보는 0점 |
| 반환에 추가하는 순위 필드 | `rank` (1부터) | [x] | 재정렬 결과 기준으로 다시 매김 |
| 기존 Dense 점수 유지 | `retrieval_score` 그대로 둠 | [x] | |

**Rerank 방식** (하나 체크):
- [x] LLM 1회로 후보 전체 채점
- [ ] LLM을 후보마다 호출 (후보 20개면 20회)
- [ ] 전용 Rerank API (MonoRouter)
- [ ] 로컬 모델 (모델 이름: ___________ , 예: `BAAI/bge-reranker-v2-m3`)
- [ ] 기타: ___________

| 항목 | 작성 |
|---|---|
| Rerank 1회당 MonoRouter 호출 수 | 1회 (채팅 모델 1회, 후보 20개를 한 프롬프트에서 JSON으로 채점) |
| 사용하는 모델 ID | ⚠ 확인 필요: `LLM_MODEL` 환경변수 (코드 기본값 `gpt-5.4-mini`, `.env` 실제 값 미확인). `common/ai_model.py`의 `get_llm_model()` 사용 |
| 추가로 설치하는 패키지 | 없음 |
| 호출 실패 시 동작 (예: Dense 순서 그대로 반환 / 예외) | **수정 완료 (로컬)**: 호출·파싱 실패 시 예외를 로그로 남기고 Dense 순서 그대로 `candidates[:final_k]` 반환. 이때 `rerank_score`는 `None`, `rank`는 Dense 순서 기준 (평가에서 폴백 여부 구분 가능) |

> 왜 필요한가: Rerank 방식에 따라 평가 1회의 호출 수가 약 18회(로컬)에서 400회 이상(후보마다 LLM)까지 달라집니다.
> MonoRouter는 분당 30회 제한이고 키를 여러 명이 함께 쓰고 있습니다.

## 2. 답변 생성

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 파일 | `generation/answer.py` | [x] | |
| 함수 이름 | `generate_answer` | [x] | |
| 입력 | `generate_answer(question: str, chunks: list[dict])` | [x] | |
| 반환 형식 | `{"answer": str, "is_answerable": bool, "used_chunk_ids": list[str]}` | [ ] | **수정 완료 (로컬)**: 기존 `answer`·`sources`는 유지하고 `is_answerable`(bool)과 `used_chunk_ids`(사용한 청크의 payload `chunk_id` 목록)를 추가. `/ask` 응답에는 현재 `answer`·`sources`만 내보내고 새 필드는 노출하지 않음 (노출 여부는 7장 C 요청 ②) |
| 출처 | 답변에 **실제로 사용한** 청크만 (검색 결과 전체 아님) | [x] | LLM이 돌려준 `used` 번호의 청크만 담음 |
| 답변 불가 처리 | 근거가 없으면 `is_answerable=False` + "문서에서 확인하기 어렵다" | [x] | "제공된 법률 문서에서는 해당 내용을 확인하기 어렵습니다."와 빈 `sources`·빈 `used_chunk_ids`, `is_answerable=False`를 반환 |
| LLM 호출 수 | 답변 1개당 1회 | [x] | 입력 청크가 비어 있으면 0회 |

출처를 어떻게 정하는지 (하나 체크):
- [x] LLM이 사용한 청크 번호·ID를 구조화 출력으로 돌려줌
- [ ] LLM이 인용한 조문 번호(예: "제2조")로 청크를 찾음
- [ ] 검색된 청크 전체를 출처로 붙임
- [ ] 기타: ___________

(LLM이 JSON `{"can_answer", "answer", "used": [근거 번호]}`로 응답하고, 코드가 번호를 청크로 되돌립니다.)

## 3. `/ask` 검색 흐름

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 지원 모드 | baseline / rerank / multi_query / combined | [ ] | 현재 `baseline`, `rerank`만 지원 (`retrieval/pipeline.py`의 `MODES`). 팀 공유 `contracts.py`의 `dense`는 `baseline`의 별칭으로 받음. `multi_query`, `combined`는 C의 `multi_query.py`를 받은 뒤 추가 예정 |
| **최종 서비스 모드** (사용자에게 기본으로 쓰는 모드) | 미정 | | ⚠ 확인 필요: 미정. 현재 기본값은 `baseline` (`RETRIEVAL_MODE` 환경변수). 평가 결과로 팀이 결정 |
| Multi Query 함수 | C의 `retrieval/multi_query.py` `multi_query_retrieve` 사용 | [ ] | 아직 연결 안 함 (파일이 저장소에 없음). C가 전달하면 `pipeline.py`에 모드로 연결 |
| combined 순서 | 확장 질의 → 질의별 Dense → RRF 병합 → **병합 후 Rerank 1회** | [ ] | 미구현. 이 순서로 구현하는 데 이견 없음 |
| 질의별 Dense 후보 수 | 20 | [x] | `CANDIDATE_K = 20` |
| 최종 K (답변 컨텍스트 청크 수) | 5 | [x] | `FINAL_K = 5` |
| 확장 질의 임베딩 | 여러 질의를 한 번의 요청으로 묶음 | [ ] | 현재 `embed_query`를 질의마다 호출. Multi Query 연결 시 묶음 임베딩으로 변경 예정 |
| 모드 선택 방법 | 내부 설정 (요청 형식은 `{"question": ...}` 그대로) | [x] | `RETRIEVAL_MODE` 환경변수 (서버 시작 시 1회 읽음) |

**공유 함수가 있는지**:
- [x] 있음 → 파일·함수: `retrieval/pipeline.py` `retrieve(question, mode="baseline", candidate_k=20, final_k=5) -> list[dict]` (`/ask`가 이미 이 함수를 사용)
- [ ] 없음 / 만들 수 있음
- [ ] 없음 / 만들 계획 없음

> 왜 필요한가: 평가의 combined와 `/ask`가 같은 흐름이어야 평가 수치가 실제 서비스를 대표합니다.
> 공유 함수가 있으면 평가가 그 함수를 그대로 호출하도록 C가 바꾸겠습니다.

## 4. Dense·Qdrant (현재 받은 코드 기준 확인)

| 항목 | C 예상 (현재 `retrieval/dense.py`) | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 함수 | `dense_retrieve(question, candidate_k=20, collection="law_articles")` | [x] | `collection` 기본값은 `QDRANT_COLLECTION` 환경변수, 없으면 `data_ingestion_law21311_v1` |
| 반환 필드 | payload 전체 + `point_id`, `retrieval_score`, `rerank_score=None`, `rank` | [x] | |
| 기본 컬렉션 | `law_articles` | [ ] | 실제: 팀 공유 스냅샷 컬렉션 `data_ingestion_law21311_v1` (46청크, `common/config.py` 기본값). `QDRANT_COLLECTION` 환경변수로 변경 가능 |
| 질의 여러 개 묶음 임베딩 함수 | 없음 (추가 요청: 예 `embed_queries(texts)`) | [x] | 현재 없음. Multi Query 연결 시 함께 추가 예정 (`embed_documents`로 한 번에 요청) |

## 5. 공통 파일·설정

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| LLM 함수 | `common/ai_model.py` `get_llm_model()` | [x] | `model`·`api_key` 인자를 받지만 실제로는 `common/config.py`의 `MODEL`·`API_KEY`를 사용함 |
| 임베딩 함수 | `common/ai_model.py` `get_embedding_model()` | [x] | 모델은 `EMBEDDING_MODEL` (기본 `text-embedding-3-small`) |
| 환경변수 이름 | `LLM_API_KEY`, `LLM_BASE_URL` (계획서는 `MONOROUTER_*`) | [x] | 추가로 `LLM_MODEL`, `EMBEDDING_MODEL`, `QDRANT_URL`, `QDRANT_COLLECTION` 사용 |
| LLM·임베딩 `max_retries` | 기본값 (429 시 2회 재시도) → 0~1로 낮출지 | [x] | 현재 기본값(설정 없음). 분당 30회 제한이 있어 **1로 낮추는 것을 제안**, C와 합의 후 적용 |
| Qdrant 주소 | `.env`의 `QDRANT_URL` (A 서버로 바꿀 예정) | [x] | |
| Qdrant API 키 지원 | 없음 (A가 Cloud·보안 설정 시 `common/qdrant.py`에 추가 필요) | [x] | 현재 없음 |

## 6. FastAPI

| 항목 | C 예상 | 같음 | 실제 (다르면 작성) |
|---|---|---|---|
| 앱 파일 | `app/main.py` | [x] | |
| `/ask` 입력 | `{"question": "..."}` | [x] | 빈 문자열은 400 |
| `/ask` 출력 | `{"answer": "...", "sources": [{"article": "제○조", "content": "..."}]}` | [x] | 모델·검색 호출 실패 시 502 |
| 평가 결과 라우터 연결 | `from evaluation.api import router` + `app.include_router(router)` (`HANDOFF_B.md` 2장) | [ ] | 실제: `from app.eval_api import router as eval_router` + `app.include_router(eval_router)` 로 이미 연결됨 (`evaluation.api` 아님) |
| 평가 결과 화면 | 만들 예정 / 안 만듦 / 미정 | | ⚠ 확인 필요: 미정. 현재는 채팅 화면(`/chat`)과 `app/viewer.py` 라우터만 있음 |

`/ask` 응답 예시를 하나 붙여 주세요 (실제 또는 예상):

```json
{
  "answer": "인공지능 기본계획은 3년마다 수립해야 합니다. 과학기술정보통신부장관이 관계 중앙행정기관의 장 및 지방자치단체의 장의 의견을 들어 3년마다 수립·변경·시행하도록 되어 있습니다. 다만, 기본계획 중 대통령령으로 정하는 경미한 사항을 변경하는 경우에는 예외입니다.",
  "sources": [
    {
      "article": "제6조",
      "content": "제6조(인공지능 기본계획의 수립) ① 과학기술정보통신부장관은 관계 중앙행정기관의 장 및 지방자치단체의 장의 의견을 들어 3년마다 … (이하 생략)"
    }
  ]
}
```

위 예시는 2026-10-08 `RETRIEVAL_MODE=rerank`로 서버를 띄워 질문 "인공지능 기본계획은 몇 년마다 수립해야 하나요?"를 실제 호출한 결과입니다 (`sources[].content`는 길어서 일부만 표기). 컬렉션은 팀 공유 스냅샷 `data_ingestion_law21311_v1`(46청크)입니다.

## 7. 일정·기타

| 항목 | 작성 |
|---|---|
| Rerank 코드 공유 예정 시점 | ⚠ 확인 필요: 미정 (로컬에서 수정 완료, 아직 안 올림) |
| 답변·`/ask` 코드 공유 예정 시점 | ⚠ 확인 필요: 미정 (로컬에서 수정 완료, 아직 안 올림) |
| C에게 요청하는 것 | 아래 "C에게 부탁할 것" 참고 |
| 기타 메모 | rerank는 점수가 누락된 후보를 0점으로 처리하고 동점이면 Dense 순서를 유지함. 답변은 마크다운 없는 평문 2~3문장으로 생성 |

## C에게 부탁할 것

1. **`retrieval/multi_query.py` 전달**: `multi_query_retrieve`의 시그니처(입력·반환)를 알려 주세요. 받으면 `retrieval/pipeline.py`의 `MODES`에 `multi_query`·`combined`를 추가하고, `/ask`와 평가가 같은 `retrieve()`를 쓰게 하겠습니다.
2. **새 필드 노출 여부 결정**: `generate_answer`가 `is_answerable`, `used_chunk_ids`를 반환하도록 바꿨습니다. 평가가 `generate_answer`를 직접 호출하면 충분한지, `/ask` 응답에도 필요한지 알려 주세요. 현재 `/ask`는 `answer`·`sources`만 반환합니다.
3. **`max_retries` 합의**: MonoRouter 분당 30회 제한과 키 공유 때문에 LLM·임베딩 `max_retries`를 1로 낮추는 것을 제안합니다. 동의하면 `common/ai_model.py`에 적용하겠습니다.
4. **rerank 폴백 처리 확인**: rerank가 실패하면 Dense 순서로 대체하고 `rerank_score=None`으로 둡니다. 평가에서 `rerank_score`가 `None`인 결과를 어떻게 다룰지(제외 / 별도 집계) 알려 주세요.
5. **평가 라우터 경로 확인**: 실제 경로는 `app/eval_api.py`(`from app.eval_api import router as eval_router`)입니다. `HANDOFF_B.md`의 `evaluation.api` 안내를 이 경로로 정정해 주세요.
