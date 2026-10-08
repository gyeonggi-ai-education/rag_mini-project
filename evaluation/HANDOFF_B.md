# C → B 전달: 평가 결과 API 연결과 인터페이스

> **상태: 초안 (더미 데이터 기준)**. A의 Qdrant 연결 후 ⚠️ 표시한 부분을 실제 값으로 바꿔 최종 공유한다.
> 갱신 절차는 `evaluation/TODO_C.md` 3장 참고.

C(Multi Query·평가)가 만든 평가 결과를 B의 FastAPI에 붙이는 방법과, 서로 맞춰야 할 함수 형식을 정리했다.
C 쪽 파일은 모두 `evaluation/`, `retrieval/multi_query.py`에 있고, B 소유 파일(`app/`, `retrieval/dense.py` 등)은 고치지 않았다.

## 1. 요약

| 구분 | 내용 |
|---|---|
| B가 할 일 | `app/main.py`에 라우터 2줄 추가 (2장) |
| B가 맞춰 줄 것 | `rerank`, `generate_answer` 함수 형식 (5장), combined 흐름과 최종 서비스 모드 (5-1장) |
| 화면에 보여 줄 것 | 모드별 검색 지표, 답변 지표, 유형별 지표, 질문별 비교, 청크별 역할 (4장) |
| 주의 | MonoRouter 분당 30회 제한, 같은 키를 여러 명이 공유 → Rerank 방식 요청 (6장) |

평가는 터미널에서 실행하고, API는 **저장된 결과 파일만 읽는다**. API 요청이 LLM을 호출하지 않는다.

## 2. FastAPI 연결

```python
# app/main.py
from evaluation.api import router as eval_router

app.include_router(eval_router)
```

- 라우터를 import해도 LLM·Qdrant·`retrieval` 모듈을 불러오지 않는다 (확인함).
- Qdrant는 청크 원문을 요청할 때(`with_source=true`, `/eval/chunks`)만 연결한다.
- 결과·문항이 없으면 404, Qdrant 연결 실패는 503으로 응답한다.
- `evaluation/run_eval.py`는 API에서 import하지 않는다 (평가 실행용, LLM 설정을 불러옴).

## 3. 엔드포인트

| 메서드·경로 | 설명 | 예시 |
|---|---|---|
| `GET /eval/runs` | 평가 실행 목록 (최신순) | `[{"stamp": "20261008_132903", "modes": ["baseline", "rerank", ...]}]` |
| `GET /eval/results?stamp=` | 실행 결과 전체. `stamp` 생략 시 최신 | 4장 구조 |
| `GET /eval/results/questions/{id}?stamp=&with_source=` | 문항 하나. `with_source=true`면 청크 원문 전체 포함 | `/eval/results/questions/D03?with_source=true` ⚠️ 실제 골든셋은 id가 1~10 |
| `GET /eval/chunks?ids=...&collection=` | chunk_id로 원본 청크 원문 조회 | `/eval/chunks?ids=d34421c4c405dba8...` (A의 chunk_id는 64자리 해시, 예는 제33조) |

## 4. 응답 구조와 화면 표시

`GET /eval/results` 응답의 최상위 필드:

```text
{
  "stamp": "20261008_132903",
  "config":   { 골든셋, 데이터 버전, 컬렉션, 모델, K, 코드 버전 ... },
  "modes":    ["baseline", "rerank", "multi_query", "combined"],
  "metrics":  { 모드: 고정 이름 지표 }     ← 화면 표시는 이것을 사용
  "summary":  { 모드: 원본 요약 }          ← 지표 이름에 K가 붙음 (hit@5 등)
  "questions": [ 질문별 결과 ]
}
```

### 4-1. 결과표 ① 모드별 검색 지표 (`metrics[모드]`)

| 표 열 | 필드 | 설명 |
|---|---|---|
| 모드 | 키 | baseline, rerank, multi_query, combined |
| 상태 | `status` | `ok` / `not_connected` (`message`에 사유, 예: Rerank 미연결) |
| Hit@1 | `hit_at_1` | 1위 청크가 정답인 질문 비율 (답변에 가장 크게 쓰이는 근거가 맞는지) |
| Hit@3 | `hit_at_3` | 상위 3개 안에 정답이 있는 질문 비율 (K가 3보다 클 때) |
| Hit@K | `hit` | 정답 근거를 하나 이상 찾은 질문 비율 |
| Recall@K | `recall` | 질문별 필요한 정답 근거 중 찾은 비율의 평균 |
| MRR | `mrr` | 첫 정답 순위의 역수 평균 (1위=1, 2위=0.5) |
| Rerank 전 후보 Recall | `candidate_recall` | rerank·combined만. 낮으면 후보 누락, 높은데 최종이 낮으면 순위 문제 |
| 평균 지연(초) | `avg_latency_sec` | 검색 단계 |
| MonoRouter 요청 수 | `monorouter_requests` | 이 모드에서 실제로 보낸 요청 수 (캐시 적중 제외) |
| 질의 확장 실패 | `expansion_failures` | 0이 아니면 Multi Query 결과를 믿기 어려움 |
| 토큰 | `total_tokens` | LLM 토큰 합 (임베딩 토큰 제외) |

`status != "ok"`인 모드는 다른 값이 없으므로 "⏸ 아직 연결되지 않았습니다"처럼 `message`만 표시한다.

### 4-2. 결과표 ② 답변 지표 (`metrics[모드].answer`, 답변을 생성한 모드만)

API 호출을 줄이려고 답변은 기본적으로 **baseline과 multi_query 두 모드만** 생성한다 (`config.answer_modes`).
나머지 모드는 `answer`가 없으므로 "답변 생성 안 함"으로 표시한다.

| 표 열 | 필드 | 설명 |
|---|---|---|
| 정답 근거 사용률 | `used_gold_rate` | 답변이 실제로 정답 청크를 근거로 썼는지 |
| 오거절률 | `false_refusal_rate` | 답변 가능한 질문을 "확인하기 어렵다"고 한 비율 |
| 답변 불가 거절률 | `unanswerable_refusal_rate` | 답변 불가 질문을 제대로 거절한 비율 |
| 필수 요소 포함률 | `avg_key_point_coverage` | 규칙 채점 (API 호출 없음). 골든셋 필수 요소가 답변에 들어간 비율 (글자 겹침 근사치) |
| 근거 겹침 | `avg_grounded_ratio` | 규칙 채점. 답변 문장이 사용한 근거 원문과 겹치는 비율 |
| LLM 정답성 / 관련성 / 충실성 | `avg_correctness` / `avg_relevance` / `avg_faithfulness` | `--judge llm`일 때만. 1~5 |
| LLM 근거 없는 주장 비율 | `unsupported_claim_rate` | `--judge llm`일 때만 |
| 생성·전체 지연 | `avg_generation_sec`, `avg_total_sec` | |

질문별 답변(`questions[].modes[모드].answer`)의 채점 필드:
- `judgement`: 규칙 채점 `{key_point_coverage, missed_key_points, grounded_ratio, ungrounded_sentences}`
  또는 `{refused_correctly}`(답변 불가 문항), `{false_refusal: true}`(답변 가능한데 거절)
- `llm_judgement`: LLM 채점 `{correctness, relevance, faithfulness, unsupported_claims, reason}` (`--judge llm`일 때만)

### 4-3. 결과표 ③ 질문 유형별 (`metrics[모드].by_type`)

유형: 기준 / MQ(일상어) / RR(경쟁 조항) / 둘다. 각 `{n, hit_at_1, hit, recall, mrr}`.
"Multi Query는 MQ 유형에서, Rerank는 RR 유형에서 효과가 있었는가"를 보여 주는 표.

### 4-4. 질문별 비교 (`questions[]`)

```text
{
  "id": "D03", "type": "MQ", "question": "...", "answerable": true,
  "gold": ["v1:제2조-5호"] 또는 ["제2조"],
  "modes": {
    "baseline": {
      "status": "ok", "rr": 1.0, "verdict_vs_baseline": null,
      "retrieved": [{"rank", "chunk_id", "article", "score", "rerank_score", "is_gold", "content(80자)"}],
      "answer": {"answer", "is_answerable", "used_chunks": [...], "judgement": {...}}
    },
    "multi_query": {
      ..., "verdict_vs_baseline": "개선" | "악화" | "동일",
      "queries": ["원질문", "확장1", ...],
      "per_query": [{"query", "chunk_ids", "gold_rank"}],
      "mq_change": {"original", "final", "added", "added_gold", "dropped"}
    },
    "rerank": {"status": "not_connected", "message": "..."}
  },
  "chunks": [ 청크별 역할 ]
}
```

화면 구성 예:
1. 질문, 정답 조문
2. 모드별 한 줄: RR, baseline 대비 판정, 검색 청크(✅/❌)
3. Multi Query: 확장 질의 목록과 질의별 정답 순위, "원질문 결과 → 최종 결과" 변화 (`mq_change`)
4. 모드별 답변과 **답변이 실제 사용한 청크** (`answer.used_chunks`)

### 4-5. 청크별 역할 (`questions[].chunks`)

```json
{"chunk_id": "v1:제2조-1호", "article": "제2조", "is_gold": false,
 "modes": {"multi_query": ["검색 2위", "확장 질의로 추가"]}}
```

역할 태그: `검색 N위`, `답변 사용`, `확장 질의로 추가`, `원질문 결과였으나 밀려남`, `Rerank 전 N위`,
`정답인데 어떤 모드도 검색 못 함`(키 `all`). 정답 → 답변 사용 → 나머지 순으로 정렬되어 있다.
`with_source=true`로 문항을 조회하면 각 청크에 `content`(원문 전체), `article_title`, `source_found`가 추가된다.

## 5. B와 맞춰야 할 함수 형식

C의 평가 코드는 아래 형식을 기대한다. 이름이나 인자가 다르면 알려 주면 C 쪽(`evaluation/run_eval.py`) 한 줄을 고친다.

| 함수 | 위치 (계획서 기준) | 입력 | 반환 |
|---|---|---|---|
| `dense_retrieve` | `retrieval/dense.py` | `(question, candidate_k, collection=)` | 청크 dict 목록 + `retrieval_score`, `rerank_score=None`, `rank` (현재 코드 그대로 사용 중) |
| `rerank` | `retrieval/rerank.py` | `(question, candidates, final_k)` | 청크 dict 목록. 입력 필드 유지 + **`rerank_score`**, **`rank`(1부터)** |
| `generate_answer` | `generation/answer.py` | `(question, chunks)` | `{"answer": str, "is_answerable": bool, "used_chunk_ids": [chunk_id, ...]}` |

- `rerank` 결과에 `chunk_id, article, content, retrieval_score, rerank_score, rank`가 없으면 평가가 필드 이름을 알려 주며 멈춘다.
- `retrieval/rerank.py`가 없으면 rerank·combined는 `not_connected`로 기록된다. 파일을 넣으면 같은 명령으로 바로 실행된다.
- `generate_answer`의 `used_chunk_ids`는 답변에 **실제로 사용한** 청크만 넣는다 (계획서 4장: 출처는 답변에 사용한 근거). C의 임시 구현(`evaluation/temp_generate.py`)은 컨텍스트에 청크 번호를 붙이고 사용한 번호를 받아 chunk_id로 되돌린다.
- 청크 필드는 `chunk_id, article, content`가 필수, `article_title, paragraph, is_supplementary`는 있으면 사용한다.
  A의 실제 payload: `chunk_id`(64자리 해시), `article`, `article_title`, `content`, `page_start`·`page_end`(문자열), `source_spans` 등.
  `paragraph`·`is_supplementary`는 없다 (조 단위 청킹). 화면에는 해시 대신 `questions[].chunks[].label`(조 번호)을 표시한다.

`/ask`에서 Multi Query를 쓸 때:

```python
from retrieval.multi_query import multi_query_retrieve

candidates = multi_query_retrieve(question, candidate_k=20)        # 기본 dense_retrieve 사용
# 임베딩을 한 번에 묶고 싶으면 prefetch=질의목록을_한번에_임베딩하는_함수 전달 (6장)
```

### 5-1. Multi Query + Rerank(combined) 흐름 맞추기

평가의 `combined` 모드가 B의 "Multi Query + Rerank" `/ask`와 **같은 흐름**이어야 평가 수치가 서비스를 대표한다.

| 항목 | C 평가의 combined | B와 확인할 것 |
|---|---|---|
| 순서 | 확장 질의 → 질의별 Dense → RRF 병합 → **병합 후 Rerank 1회** | 질의마다 Rerank 후 병합하는 방식이면 알려 줄 것 |
| Rerank 입력 후보 수 | 20 (`RERANK_INPUT_K`) | |
| 질의별 Dense 후보 수 | 20 | |
| 최종 K | 5 (계획서 초기값) | |

제안: B가 `/ask`용으로 `retrieve(question, mode)`(mode: baseline/rerank/multi_query/combined) 함수를 만들면,
C의 평가가 그 함수를 그대로 호출하도록 바꾼다. 평가와 서비스가 같은 코드를 쓰게 되어 어긋나지 않는다.

**최종 서비스 모드를 정해 달라.** 평가의 답변 비교는 "baseline vs 최종 서비스 모드" 두 개만 생성한다
(호출 절감). 지금 기본값은 baseline·multi_query이고, combined로 서비스하면 baseline·combined로 바꾼다.

## 6. MonoRouter 호출 수 (중요)

- MonoRouter는 **분당 30회** 제한이고, `.env`의 키가 "universal api key"라 **여러 명이 같은 한도를 나눠 쓰는 것으로 보인다.** C가 분당 25회로 제한했는데도 429가 났다.
- **임베딩도 MonoRouter 요청이다.** Dense 검색 1번 = 요청 1번.
- 429를 받으면 LLM 클라이언트가 기본 2번 재시도해서 요청이 더 늘어난다.

### C 평가의 호출 수 (실제 골든셋 10문항, 4개 모드, Rerank 1회 방식 가정)

| 단계 | 원래 | 현재 | 방법 |
|---|---|---|---|
| 임베딩 | 120 | **1** | 모드끼리 공유 + 질문·확장 질의 전체를 한 번의 요청으로 |
| 질의 확장 | 20 | **1** | 10문항을 한 번의 요청으로, combined는 재사용 |
| Rerank (B) | 20 | 20 | 같은 후보면 캐시 (재실행 0) |
| 답변 생성 | 40 | **약 16** | baseline·multi_query 두 모드만, 같은 청크면 재사용 |
| 채점 | 40 | **0** | 규칙 기반 (LLM 채점은 선택, 문항당 1회 = 10) |
| **합계** | **220** | **약 38 (−83%)** | 같은 평가 재실행은 **0** |

- 개발 중 검색만 평가할 때: 원래 60회 → **2회** (−97%)
- 노트북: 저장된 결과만 읽어서 **0회** (외부 네트워크를 막고 실행해서 확인함)
- 수치는 코드 구조와 더미 결과로 계산한 값이다 (답변 재사용 비율은 더미 12문항 중 5문항에서 baseline과 multi_query의 최종 청크가 같았던 것을 적용).
  ⚠️ 실제 데이터 평가 후 결과의 `monorouter_requests` 실측값으로 교체.

### 질문 하나(`/ask`)의 호출 수

| 단계 | 호출 |
|---|---|
| 질의 확장 (Multi Query) | LLM 1 |
| 임베딩 | 원질문 + 확장 질의를 **묶으면 1**, 따로 하면 4 |
| Rerank | 후보 전체를 **한 번에 하면 1**, 후보마다 하면 20, 로컬 모델이면 0 |
| 답변 생성 | LLM 1 |
| **합계** | **3~4회 (약 5~10초)**. 묶지 않으면 25회 이상 |

### B에게 요청

1. **Rerank는 후보 전체를 한 번의 호출로**, 가능하면 **로컬 Rerank 모델**(예: `bge-reranker` 계열 cross-encoder)로.
   남은 호출 38회 중 20회가 Rerank다. 로컬 모델이면 최종 평가가 약 18회로 1분 안에 끝난다.
   후보마다 LLM을 호출하면 다른 절감을 다 해도 평가 1회에 400회 이상이 된다.
2. `retrieval/dense.py`에 **질의 여러 개를 한 번에 임베딩하는 함수**(예: `embed_queries(texts)`)를 두고,
   `/ask`에서 `multi_query_retrieve(question, prefetch=embed_queries 를 캐시에 넣는 함수)`로 쓰면 임베딩 4회 → 1회.
   평가에서는 C가 실행 중에만 `dense._embedding`을 캐시 객체로 감싸서 처리하고 있다 (파일은 수정하지 않음).
3. `common/ai_model.py`의 LLM·임베딩 `max_retries`를 0~1로 낮추는 것을 검토 (429 재시도가 요청을 늘림). 공통 파일이라 B가 결정.
4. 최종 평가는 `/ask` 시연·테스트와 **같은 시간에 돌리지 않는다** (같은 키의 분당 한도를 나눠 쓰므로).

## 7. 평가 실행 (참고)

A의 데이터는 스냅샷으로 공유되어 각자 로컬 Qdrant에 `data_ingestion_law21311_v1`로 복구한다 (`rag-team-share/docs/TEAM_ONBOARDING.md`).
C 평가는 `--collection`을 생략하면 이 컬렉션을 쓴다 (`QDRANT_COLLECTION` 환경변수로 변경 가능, B의 `dense.py`와 같은 변수).

```bash
# 개발 중: 검색만 (첫 실행 2회, 재실행 0회)
uv run python -m evaluation.run_eval --k 5 --data-version <버전>
# 최종: 답변 baseline·multi_query, 규칙 채점 (약 38회, 재실행 0회)
uv run python -m evaluation.run_eval --k 5 --answer shared --data-version <버전>
# LLM 채점까지 (+문항당 1회): --judge llm / 답변 모드 변경: --answer-modes baseline rerank multi_query
# 결과 확인 (API 호출 없음)
uv run python -m evaluation.results_view --detail
uv run python -m evaluation.report              # evaluation/results/<stamp>_report.md
```

- 결과 파일: `evaluation/results/<stamp>_<mode>.json`, 호출 캐시: `evaluation/.cache/` (둘 다 git 제외)
- 결과의 `config.prepare`에 묶음 처리 정보, `summary.monorouter_requests`에 모드별 실제 요청 수가 남는다.
- 더미 데이터 결과(`golden_set.dummy.jsonl`, 컬렉션 `tmp_c_law_articles`)는 연결 확인용이다. 전체 문서 성능으로 표시하지 않는다.
- `config.data_version`이 `dummy-v1`이면 화면에 "더미 데이터"라고 표시하는 것을 권장한다.

## 8. 실제 데이터 평가 결과 ⚠️ A 연결 후 작성

| 항목 | 값 |
|---|---|
| Qdrant 주소·컬렉션 | 로컬 `http://localhost:6333` · `data_ingestion_law21311_v1` (A 스냅샷 복구, 46개 청크) |
| 데이터 버전 | 법률 제21311호 (시행 2026. 7. 21.), 청킹 `whole-article-v1` (조 단위) |
| 평가 실행 시각(stamp) | (실행 후) |
| 모드별 Hit@5 / MRR | (실행 후, `results_view` 표 복사) |
| 실측 MonoRouter 요청 수 | (실행 후) |
| 대표 실패 사례 | (실행 후, 질문별 목록에서 선택) |
