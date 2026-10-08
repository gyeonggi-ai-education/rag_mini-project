# B 회신: C 회신(`c to b_2.md`) 확인 결과와 반영 내용

`c to b_2.md` 잘 받았습니다. C가 요청한 코드 수정은 반영했고, 물어보신 `app/eval_api.py`는 아래에 답했습니다.
코드는 아직 커밋·푸시 전이라 B 로컬 작업 폴더에만 있습니다.

- 작성자: B
- 작성일: 2026-10-08
- 코드 위치: 아직 안 올림 (로컬, `main` 브랜치 미커밋)

---

## 1. 반영 완료 (로컬에서 동작 확인함)

| 요청 | 반영 내용 | 확인 |
|---|---|---|
| ③ `max_retries` 1 | `common/config.py`에 `MAX_RETRIES`(환경변수 `LLM_MAX_RETRIES`, 기본 1)를 추가하고 `common/ai_model.py`의 LLM·임베딩 둘 다에 적용 | 두 객체 모두 `max_retries == 1` |
| ④ 실패 원인 `rerank_error` | Rerank 실패 시 반환하는 모든 청크에 `rerank_error: "<예외 클래스 이름>"` 추가 (예: `RateLimitError`). `rerank_score`는 전부 `None`, `rank`는 Dense 순서 | LLM 호출을 강제로 실패시켰을 때 5개 모두 `None` + `RuntimeError` 기록 |
| 추가 부탁 1: `retrieve()`의 `stats` | `retrieve(question, mode, candidate_k=20, final_k=5, stats=None)`. 넘기면 `stats["candidates"]`에 Rerank 전 후보를 담음. `/ask`는 넘기지 않아 동작 변화 없음 | 후보 20개 기록 확인 |
| ① 묶음 검색 | `retrieval/dense.py`에 추가 (아래 참고) | 질의 2개를 임베딩 요청 1회로 처리, 단건 검색과 결과 일치 |

**묶음 검색 함수** (C가 제안한 `dense_many_fn` 형태 `Callable[[list[str], int], list[list[dict]]]`에 맞춤)

```python
embed_queries(texts: list[str]) -> list[list[float]]
# 여러 질의를 임베딩 1회 요청으로 벡터화. 입력 순서 유지.

dense_retrieve_many(questions: list[str], candidate_k=20, collection=COLLECTION_NAME) -> list[list[dict]]
# 임베딩 1회 + Qdrant 1회(query_batch_points). 질의별 결과 목록 반환.
# 각 청크는 dense_retrieve와 같은 필드: payload 전체 + point_id, retrieval_score, rerank_score=None, rank
```

`multi_query_retrieve(..., dense_many_fn=dense_retrieve_many)`로 넘기면 됩니다. 컬렉션은 기본값을 쓰므로 `collection` 인자는 따로 넘기지 않아도 됩니다.

## 2. ⑤ `app/eval_api.py` 확인 결과

**독립 구현입니다.** `evaluation/results_view.py`를 불러다 쓰지 않습니다.

- `evaluation/results/` 아래의 JSON을 직접 읽습니다 (`rglob("*.json")`).
- 제공하는 경로: `GET /eval`(화면), `GET /api/eval/results`(목록·요약), `GET /api/eval/results/{file}`(파일 내용).
- 목록은 결과 JSON의 `experiment`, `config.created_at`, `config.data_version`, `counts`, `metrics.overall`, `cost.latency_sec`를 읽습니다. 화면(`app/static/eval.html`)은 각 문항의 `retrieved[].rank/article/title/point_id/…`를 읽습니다.
- 이 저장소에는 `evaluation/results_view.py`, `evaluation/api.py`가 없습니다.

즉 C가 말한 두 번째 경우(같은 기능이 두 벌)입니다. **B는 C의 `evaluation/results_view.py`를 쓰도록 맞추는 데 동의하고, 1안으로 진행하려 합니다.**
결과 JSON 형식이 바뀌면 B 화면이 깨질 수 있으므로, 합치기 전에 필드 대조를 하겠습니다.

1. B의 `eval_api.py`를 `results_view.py`의 `list_runs`, `get_results`, `get_question`을 호출하는 얇은 라우터로 바꾸기 (B가 수정, C가 함수 시그니처 전달)
2. 당분간 둘 다 두되, 결과 JSON의 필드(위 목록)는 유지하기

## 3. 데이터·이름 변경 알림 (C 평가에 영향 있을 수 있음)

1. **데이터 교체**: 팀 공유 스냅샷 `data_ingestion_law21311_v1`(46청크)을 복구했고, `common/config.py`의 기본 컬렉션을 이것으로 바꿨습니다. 기존 `law_articles`(3개 샘플)는 그대로 남아 있습니다. C 평가와 골든셋도 이 컬렉션 기준으로 확인이 필요합니다.
2. **조문 제목 필드**: 이 컬렉션 payload에는 `title`이 없고 `article_title`이 있습니다. B 코드는 `article_title`을 먼저 쓰고 없으면 `title`을 씁니다. C 코드가 `title`을 읽고 있으면 알려 주세요.
3. **chunk_id**: payload에 `chunk_id`가 있어 `used_chunk_ids`와 chunk_id 기준 중복 제거에 그대로 쓸 수 있습니다.
4. **모드 이름**: 팀 공유 `common/contracts.py`는 `dense`/`rerank`/`multi_query`/`multi_query_rerank`를 씁니다. B는 `baseline`, `rerank`를 쓰고 `dense`를 `baseline`의 별칭으로 받습니다. `multi_query` 계열의 최종 이름은 평가와 `/ask`가 같은 값을 쓰도록 정해 주세요.

## 4. 아직 안 된 것

- `multi_query`, `combined` 모드: `retrieval/multi_query.py`가 저장소에 아직 없어 `pipeline.py`에 연결하지 않았습니다. 파일을 올려 주시면 `MODES`에 추가하고 `/ask`와 평가가 같은 `retrieve()`를 쓰도록 하겠습니다.
- 새 컬렉션(46청크) 기준 baseline/rerank 평가 수치: 아직 측정하지 않았습니다.
- 코드 커밋·푸시: B가 정해서 올리겠습니다. 올리면 알려 드리겠습니다.

## 5. B가 C에게 요청하는 것

1. `retrieval/multi_query.py`를 저장소에 올려 주세요.
2. ⑤ 라우터는 **1안(B 라우터가 C의 `results_view.py`를 호출)으로 가겠습니다.** `evaluation/results_view.py`는 받았습니다(B 저장소 `evaluation/`에 넣었고 import 확인함). B 화면(`eval.html`)이 결과를 읽는 형식(`experiment`, `metrics.overall`, `per_question`)이 `results_view.py`의 형식(`summary`, `rows`, 모드가 나란히 묶인 실행)과 달라서, 화면과 `eval_api.py`를 새 형식에 맞춰 다시 만들려고 합니다. 이를 위해 아래를 보내 주세요.
   - `evaluation/chunk_source.py` (`get_question(with_source=True)`가 불러오는 파일)
   - **새 형식의 샘플 결과 JSON 1~2개** (`evaluation/results/{YYYYMMDD_HHMMSS}_{mode}.json`, 가능하면 baseline·rerank 각 1개). 이게 있어야 화면을 만들고 테스트할 수 있습니다.
   - `evaluation.run_eval`과 골든셋(`golden_set.jsonl`)을 B 저장소에서 돌릴 방법. B의 `scripts/eval_retrieval.py`는 결과 형식이 달라 이 화면에 나오지 않게 되므로, 평가 실행은 C의 `run_eval`로 일원화하는 것을 제안합니다. C의 `evaluation/api.py`는 `/eval` 경로가 겹치므로 `main.py`에 등록하지 않고 B의 `eval_api.py`만 쓰겠습니다.
3. `multi_query` 계열 모드의 이름을 정해 주세요 (3장 4번).
4. 새 컬렉션 기준으로 골든셋의 정답 조문 표기(예: `제2조`, `제22조의2`)가 맞는지 확인해 주세요. B의 `scripts/eval_retrieval.py`는 컬렉션에 정답 조문이 없는 문항을 평가에서 제외합니다.
