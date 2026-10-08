# C 회신: B 구현 정보 확인 결과와 요청 답변

`b to c.md` 잘 받았습니다. 대부분 C 평가 코드가 예상한 형식과 같아서 큰 수정 없이 연결할 수 있습니다.
아래에 확인 결과, B가 부탁한 5가지에 대한 답, C가 추가로 부탁할 것을 정리했습니다.

- 작성자: C
- 작성일: 2026-10-08
- 기준: `b to c.md` 내용 (B 코드는 아직 커밋 전이라, 코드가 올라오면 다시 대조합니다)

---

## 1. 확인 결과: 그대로 연결 가능

| 항목 | 결과 |
|---|---|
| `rerank(question, candidates, final_k)`, `rerank_score`·`rank` 반환 | C 평가와 그대로 맞음 |
| Rerank 방식: LLM 1회로 후보 20개 채점 | 평가 1회 호출 수 계산(약 38회)이 그대로 유효함. 감사합니다 |
| `generate_answer` 반환에 `is_answerable`, `used_chunk_ids` 추가 | C 평가의 `--answer shared`로 바로 사용 가능 |
| 출처 = LLM이 돌려준 번호의 청크만 | C 평가의 "답변이 실제로 사용한 청크"와 같은 방식 |
| 질의별 Dense 후보 20, 최종 K 5, combined 순서(병합 후 Rerank 1회) | C 평가와 같음 |
| 모델 ID ("확인 필요"로 남긴 항목) | `.env`에 `LLM_MODEL`이 없어서 코드 기본값 `gpt-5.4-mini`가 쓰입니다. C 평가 결과에도 `gpt-5.4-mini`로 기록되어 있습니다 |

## 2. B가 부탁한 5가지에 대한 답

### ① `multi_query_retrieve` 시그니처

파일: `retrieval/multi_query.py`

```python
multi_query_retrieve(
    question: str,
    candidate_k: int = 20,          # 병합 후 반환할 후보 수
    per_query_k: int | None = None, # 질의 하나당 Dense 후보 수 (기본 candidate_k)
    dense_fn=None,                  # 기본 retrieval.dense.dense_retrieve
    llm=None,                       # 기본 common.ai_model.get_llm_model()
    stats: dict | None = None,      # 넘기면 확장 질의·질의별 결과·호출 수를 기록 (평가용)
    before_llm_call=None,           # 평가용 (호출 속도 제한)
    queries: list[str] | None = None,  # 이미 확장한 질의 (평가용 캐시), 첫 번째는 원질문
    prefetch=None,                  # 평가용 (아래 참고)
) -> list[dict]
```

- 반환: 청크 dict 목록. 입력 필드 유지 + `retrieval_score`(**RRF 점수**, Dense 유사도 아님), `rerank_score=None`, `rank`(1부터)
- 질의 확장: 원질문 포함 최대 4개 (원질문 + 확장 3개), LLM 1회. 확장 실패 시 원질문만으로 검색 (예외 없음)
- `/ask`에서는 `multi_query_retrieve(question, candidate_k=20)`만 쓰면 됩니다. 나머지 인자는 평가용입니다.
- combined는 이렇게 연결하면 C 평가와 같은 흐름입니다.
  ```python
  candidates = multi_query_retrieve(question, candidate_k=20)
  results = rerank(question, candidates, final_k=5)
  ```

**묶음 임베딩**: 지금 `prefetch`는 C 평가의 캐시 래퍼가 있어야 동작해서 `/ask`에서 바로 쓰기 어렵습니다.
B가 `embed_queries(texts)`를 만들 예정이라고 해서, C가 `multi_query_retrieve`에 아래 인자를 추가하겠습니다.

```python
dense_many_fn: Callable[[list[str], int], list[list[dict]]] | None = None
# 질의 여러 개를 받아 질의별 Dense 결과 목록을 돌려주는 함수.
# 주면 질의마다 dense_fn을 부르지 않고 이 함수를 한 번 부릅니다.
```

B 쪽 구현 예 (`retrieval/dense.py`): `embed_queries(texts)`로 벡터를 한 번에 받고, 벡터마다 Qdrant를 검색해 질의별 결과를 반환.
이름이나 형태를 다르게 하고 싶으면 알려 주세요. B 구현에 맞추겠습니다.

### ② 새 필드(`is_answerable`, `used_chunk_ids`)를 `/ask`에도 노출할지

**평가에는 필요 없습니다.** 평가는 `generate_answer`를 직접 호출합니다.
화면에서 "답변 불가"를 따로 표시하고 싶을 때만 `/ask`에 노출하면 됩니다. 결정은 B에게 맡기겠습니다.

### ③ `max_retries`를 1로 낮추기

**동의합니다.** 적용해 주세요.
C 평가는 자체적으로 분당 25회 제한과 429 시 1분 대기 후 재시도를 하고 있어서, 클라이언트 재시도는 1회면 충분합니다.

### ④ Rerank 실패(Dense 순서로 대체)를 평가에서 어떻게 다룰지

**제외하지 않고 "Rerank 실패 수"로 따로 집계하겠습니다.**

- 이유: 실패 시 서비스도 그 결과를 사용자에게 내보내므로, 평가에서 빼면 실제보다 좋게 보입니다.
- 실패한 결과는 C의 캐시에 저장하지 않고, 한 번 기다렸다가 다시 시도합니다.
- 실패 판정: 반환된 결과의 `rerank_score`가 **모두 `None`**이면 실패로 봅니다. 이 규칙을 유지해 주세요.
  (정상 결과에서 일부 후보만 점수가 없을 때는 0점으로 처리한다고 하셨으니, 정상 결과에 `None`은 없다고 이해했습니다. 다르면 알려 주세요.)
- 가능하면 실패 원인도 구분되면 좋겠습니다. 429(호출 제한)로 실패했는지 알 수 있으면 C가 1분 기다렸다 재시도하고, 다른 오류면 바로 기록만 합니다.
  예: 결과 dict에 `rerank_error: "RateLimitError"` 같은 필드. 필수는 아닙니다.

### ⑤ 평가 라우터 경로 (`app.eval_api`)

정정하기 전에 **`app/eval_api.py` 내용을 확인하고 싶습니다.**

- `evaluation/results_view.py`의 함수(`get_results`, `get_question`, `list_runs`)를 **불러다 쓰는 파일**이면 → `HANDOFF_B.md`의 경로만 `app.eval_api`로 고치겠습니다. 이 경우 C의 `evaluation/api.py`는 지우거나 참고용으로 두겠습니다.
- **새로 구현한 파일**이면 → 같은 기능이 두 벌이 되어, C가 결과 형식을 바꿀 때 B 화면이 깨질 수 있습니다. 이 경우 `evaluation/results_view.py`를 쓰도록 맞추는 것을 제안합니다.

파일 내용을 공유해 주시면 바로 정리하겠습니다.

## 3. C가 추가로 부탁할 것

1. **`retrieve()`가 중간 결과를 돌려줄 방법**
   `/ask`와 평가가 같은 `retrieval/pipeline.py`의 `retrieve()`를 쓰는 데 찬성합니다. 다만 평가는 최종 결과 외에 아래도 필요합니다.
   - Rerank 전 후보 목록 (후보 누락인지 순위 문제인지 구분)
   - Multi Query의 확장 질의와 질의별 Dense 결과 (질의별 정답 순위)

   제안: `retrieve(question, mode, candidate_k=20, final_k=5, stats: dict | None = None)`처럼 `stats`를 받아서,
   있으면 `stats["candidates"]`(Rerank 전 후보)를 채우고, Multi Query 모드에서는 `multi_query_retrieve(..., stats=stats)`로 그대로 넘겨 주세요.
   `/ask`는 `stats`를 넘기지 않으므로 동작이 바뀌지 않습니다.

2. **코드 공유 시점**
   Rerank·답변·`pipeline.py`가 커밋되면 알려 주세요. md 기준으로 맞춘 부분을 실제 코드와 대조하고, 더미 데이터로 rerank·combined 평가를 한 번 돌려 보겠습니다 (Rerank 1회 방식이라 더미 12문항 기준 약 24회 추가).

3. **`common/config.py` 공유**
   B가 추가한 `CANDIDATE_K`, `FINAL_K`, `RETRIEVAL_MODE`, `QDRANT_COLLECTION`을 C 평가도 쓰도록 맞추겠습니다. 지금은 C 쪽에 같은 값(20, 5)이 따로 있어서, B 코드가 합쳐진 뒤 한 곳으로 모으겠습니다.

4. **`retrieval/dense.py`는 B 버전이 기준**
   C 저장소에 있는 `dense.py`는 이전에 받은 버전이라 `QDRANT_COLLECTION` 반영이 없습니다. 합칠 때 B 버전을 그대로 쓰겠습니다.

## 4. C가 할 일 (B 코드와 관계없이 먼저 진행)

| 할 일 | 내용 |
|---|---|
| Rerank 실패 집계 | `rerank_score`가 모두 `None`이면 실패로 기록, 지표에 실패 수 표시, 실패 결과는 캐시 안 함 |
| 묶음 검색 인자 | `multi_query_retrieve`에 `dense_many_fn` 추가 (①) |
| 기본 컬렉션 | C의 원문 조회도 `QDRANT_COLLECTION` 환경변수를 읽도록 |
| `HANDOFF_B.md` 갱신 | Rerank 1회 확정, 답변 함수 형식, `pipeline.retrieve`, `max_retries` 합의 반영 (라우터 경로는 ⑤ 확인 후) |
