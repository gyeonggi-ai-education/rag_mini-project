# 평가 프로토콜 (팀 공통 규칙)

AI 기본법 근거 기반 QA 프로젝트에서 Rerank와 Multi Query를 **공정하게 비교**하기 위한 공통 규칙이다.
협업 계획서(A 데이터·청킹 / B Rerank·API / C Multi Query·평가)를 기준으로 작성했다.
두 방식이 서로 다른 부분은 **검색 개선 기법뿐**이어야 한다.

- 대상 문서: `data/raw/ai_basic_act.pdf` (본문 표기: 법률 제21311호, 시행 2026. 7. 21. — 원문 대조 후 확정)
- 골든셋: [`golden_set.json`](golden_set.json) (v1)
- 이 문서의 수치는 논의용 제안이며, 구현·성능 검증을 마쳤다는 뜻이 아니다. `미정`인 항목은 합의 후 채운다.

---

## 1. 역할과 디렉터리 소유권

| 담당 | 책임 | 소유 경로 |
|---|---|---|
| A | 데이터·청킹, 임베딩·Qdrant 적재 | `ingestion/` |
| B | 공통 Dense 검색, Rerank, FastAPI `/ask`, Context·답변 생성, 모델 호출 어댑터 | `retrieval/dense.py`, `retrieval/rerank.py`, `api/`, `generation/` |
| C | Multi Query, 평가 실행기·지표 계산, 실험 기록 | `retrieval/multi_query.py`, `evaluation/` |

- 공통 `schemas/`는 초기에 합의하고, 변경하면 먼저 공유한다.
- 전체 인덱스 적재는 A만 한다. B·C는 읽기 중심으로 사용한다.
- 공유 컬렉션을 덮어쓰지 않는다. 재청킹 실험은 새 컬렉션·데이터 버전으로 분리한다.

## 2. 반드시 동일해야 하는 것

| 구분 | 항목 | 값 | 상태 |
|---|---|---|---|
| 데이터 | 원본 문서 | `data/raw/ai_basic_act.pdf` | 확정 |
| 데이터 | 문서 버전 | 본문에서 확인한 표기를 `document_version`에 기록 | 확인 필요 (파일명으로 확정하지 않음) |
| 데이터 | 샘플 청크 | `data/chunks/sample_chunks.jsonl` | 확정 (A가 대표 조문을 수작업 확인해 제공) |
| 데이터 | 전체 청크 | `data/chunks/chunks_v1.jsonl` (바뀌면 v2, v3) | 제안 (파일명은 A와 확인) |
| 색인 | 임베딩 모델·차원 | 미정 (MonoRouter 지원 확인 후 확정). 잠정 후보: `text-embedding-3-small`, 1536 | 미정 |
| 색인 | 거리 함수 | COSINE | 확정 |
| 색인 | 컬렉션 | `ai_basic_act_sample`, `ai_basic_act_v1` (청크 버전과 동일하게 분리) | 제안 |
| 색인 | 컬렉션 메타 | 임베딩 모델·차원·데이터 버전을 기록 | 확정 |
| 색인 | 임베딩 설정 | 문서와 질문에 **동일 설정** 사용. A와 B는 같은 임베딩 호출 함수 공유 | 확정 |
| 평가 | 골든셋 | `golden_set.json` v1, 확정 후 수정 금지 | 확정 |
| 평가 | 평가 실행기 | C 소유, `evaluation/`. 지표 계산은 이 코드 하나만 사용 | 확정 |
| 검색 | 후보 수 `candidate_k` | 20 | 제안 |
| 검색 | 최종 K | 5 (모든 지표는 K=5 기준으로 계산) | 제안 (계획서) |
| 검색 | 확장 질의 수 | 최대 3 (원질문 포함 별도) | 제안 |
| 생성 | 답변 모델·프롬프트·temperature | 미정 (프롬프트는 파일로 공유, temperature 0 권장) | 미정 |

### 모델 호출 설정
- 환경변수: `MONOROUTER_API_KEY`, `MONOROUTER_BASE_URL`. 값은 환경에서 주입하고 저장소·노트북·보고서에 기록하지 않는다.
- 합의할 것: 답변·질의 확장용 모델 ID, 임베딩 모델 ID, Rerank 방식·모델 ID. 환경변수 이름만으로 지원 기능을 가정하지 않는다.
- 첫 15분에 **채팅, 임베딩, Rerank** 호출을 각각 소량으로 확인한다. 채팅이 되어도 임베딩·Rerank가 된다고 가정하지 않는다.
- Rerank 전용 API가 없으면 LLM 기반 재정렬로 대체한다. 이 경우 보고서에 LLM 기반임을 명시하고 지연·비용을 기록한다.
- 임베딩이 지원되지 않으면 별도 API 또는 사전 준비된 로컬 모델을 확정한다.
- 참고: 현재 `common/config.py`는 `LLM_API_KEY`, `LLM_BASE_URL`, `EMBEDDING_MODEL`을 읽는다. MonoRouter 환경변수와 어떻게 연결할지는 B(모델 호출 어댑터)가 정한다.

## 3. 청크 계약 (A ↔ B, C)

첫 10분에 필수 필드를 고정하고, 선택 필드 때문에 샘플 전달을 늦추지 않는다.

| 필드 | 필수 | 규칙 |
|---|---|---|
| `chunk_id` | O | 문서 버전·조문 경로·분할 순서로 **재현 가능**하게 생성. 구조·청킹 규칙이 바뀌면 데이터 버전을 바꾼다 |
| `document_id` | O | 문서 식별자 |
| `document_version` | O | 본문에서 확인한 법률 버전 표기. 확인 전에는 `미확정`으로 기록 |
| `article` | O | 원문 조문 표기. `제2조`, `제22조의2` 형식. **골든셋 `gold_articles`와 문자열이 완전히 같아야 한다** (앞뒤 공백, 숫자만 쓰기 금지) |
| `content` | O | 출처로 제시할 수 있는 **실제 법률 본문**. 임베딩용 제목·경로 결합 텍스트가 필요하면 별도로 만든다 |
| `law_name`, `article_title`, `paragraph`, `chapter`, `section`, `page_start`, `page_end`, `is_supplementary` | 선택 | 추출 가능한 값만 채운다. 해당 구조가 없으면 `null`. **값을 추측하지 않는다** |

- `page_start`, `page_end`: PDF 물리 페이지 기준, 1부터 시작.
- 본문과 부칙을 섞어 처리하면 `is_supplementary`(또는 동등한 필드)는 **반드시** 채운다.
- 항·호·목은 `article`에 넣지 않는다. `paragraph` 등에 둔다.
- Qdrant point ID ↔ `chunk_id` 매핑은 A가 관리한다. 적재를 재실행해도 중복되지 않아야 한다.

### 검색 결과 (`RetrievedChunk`)
청크 필드에 `retrieval_score`, `rerank_score`, `rank`를 더한다. 서로 다른 단계의 점수는 구분하며, 의미가 다른 점수를 단순 합산하지 않는다.

```
retrieve(question, candidate_k) -> list[RetrievedChunk]
  B: dense_retrieve(question, candidate_k), rerank(question, candidates, final_k)
  C: expand_queries(question), multi_query_retrieve(question, candidate_k)
```

## 4. 평가 방식

### 정답 판정 (조 단위) — 계획서에 없는 세부 규칙은 *(제안)* 으로 표시
- 검색 결과 상위 K개에서 `article`을 순서대로 뽑아 **조 단위로 중복 제거**한 뒤 `gold_articles`와 비교한다. *(제안)*
- `is_supplementary`가 `true`인 청크는 정답으로 인정하지 않는다 (골든셋 정답은 모두 본문 조문). *(제안)*
- 지표 단위는 **조(article)** 이다. 항·호 단위나 Chunk 단위 지표를 쓰려면 별도로 명시한다.

### 검색 지표
| 지표 | 정의 |
|---|---|
| Hit@K | 정답 근거 중 하나 이상을 찾은 질문 비율 |
| Recall@K | 질문별 필요한 정답 근거 중 검색된 비율의 평균 (multi 유형은 반드시 함께 본다) |
| MRR | 첫 정답 근거 순위의 역수 평균. K 안에 없으면 0 *(0 처리는 제안)* |
| Recall@candidate_k | Rerank 전 후보 풀의 정답 포함률. **후보 누락**과 **순위 문제**를 구별하기 위해 기록 (계획서: "Rerank 전 후보 Recall도 확인") |

- 전체 점수와 유형별(definition, lookup, numeric, paraphrase, multi) 점수를 모두 보고한다.
- 답변 불가 문항(`answerable: false`)은 검색 지표 평균에 **섞지 않는다.**

### 답변 평가 (실험 전에 기준 고정) — 채점 방식은 *(제안, 팀 합의 필요)*
계획서의 답변 평가 항목(정답성, 관련성, 근거 충실성, 근거 없는 주장, 답변 불가 질문 처리) 중 아래 4개를 기본 평가 기준으로 하고, 답변 불가 질문 처리는 `answerable: false` 문항에만 따로 적용한다.

| 항목 | 0/1 채점 기준 (1점 조건) | 적용 문항 |
|---|---|---|
| 정답성 | 핵심 내용이 골든셋 `reference`와 일치 | 답변 가능 문항 |
| 관련성 | 질문에 직접 답함 | 전체 |
| 근거 충실성 | 답변의 주장이 제시한 출처 조문(`sources`)으로 뒷받침됨 | 답변 가능 문항 |
| Hallucination 여부 | 출처에 없는 내용·조문 번호·수치를 지어내지 않음 (**1 = 없음**, 높을수록 좋게 통일) | 전체 |
| 답변 불가 처리 | 근거가 없는 질문에 "문서에서 확인되지 않는다"는 취지로 답하고 내용을 지어내지 않음 | 답변 불가 문항 |

- 척도는 **0/1**이다. 항목별 평균 비율과 문항별 합계를 함께 보고한다.
- **채점자: LLM 1차 채점 + 사람 확인.**
  - LLM에는 문항의 정답 조문 원문, `reference`, 답변, 출처를 함께 주고 항목별 0/1과 **근거 한 줄**을 출력하게 한다.
  - 사람이 `split: final` 문항 전부를 원문과 대조해 확인하고, `tune` 문항은 일부를 표본 확인한다. 불일치 시 **사람의 판정이 우선**이며 불일치율을 보고서에 기록한다.
- LLM 채점 설정: 프롬프트 파일 고정, temperature 0, 채점 모델 ID를 결과 파일에 기록. 가능하면 **답변 생성 모델과 다른 모델**을 쓴다 (MonoRouter에서 사용 가능한 모델 확인 후 결정).

### 운영 지표
검색·Rerank·생성·전체 지연, LLM 호출 수, 토큰 수, 가능한 경우 비용. 개선이 추가 연산에서 나온 것인지 판단하기 위해 **후보 예산과 모델 호출 수를 함께 기록**한다.

## 5. 골든셋

- 파일: `golden_set.json` — **기본 10문항**(답변 가능 8 + 답변 불가 2, `stage: base`). 4시간을 확보하면 `stage: extend` 5문항을 더해 **15문항**으로 확장한다 (계획서 7절).
- 문항은 정의, 적용 범위, 의무·예외, 여러 조문 필요, 일상 표현을 포함한다.
- 기본 10문항에서 빠진 후보 30개는 `golden_pool.json`에 보관하며 **평가에 쓰지 않는다.**
- `split`: `tune`(개발·파라미터 조정용), `final`(최종 확인용). **final은 튜닝에 쓰지 않는다.**
- 정답은 청크 ID가 아니라 **조문 번호**로 관리하므로 청킹이 바뀌어도 평가 기준이 깨지지 않는다.
- 필드: `id`, `type`, `answerable`, `split`, `question`, `gold_articles`, `gold_detail`, `reference`(필수 답변 요소 요약).
- **초안이다.** 정답 조문·항은 팀이 원문과 대조해 확정해야 한다. 소규모 탐색 결과라는 한계를 보고서에 명시한다.
- 확정 후에는 수정하지 않는다. 오류를 발견하면 `meta.version`을 올리고 팀에 알린다.

## 6. 비교 실험

| 모드 | 흐름 | 확인하려는 효과 |
|---|---|---|
| baseline | 원질문 → Dense → 최종 K | 비교 기준 |
| rerank | 원질문 → Dense 후보 → Rerank → 최종 K | 후보에 있는 근거의 순위 개선 |
| multi_query | 원질문 + 확장 질의 → Dense → 병합 → 최종 K | 표현 차이로 놓친 근거 회수 |
| combined | Multi Query → 병합 후보 → Rerank → 최종 K | 두 개선의 결합 효과 (시간 여유 시) |

- Multi Query는 원질문을 반드시 포함하고 확장 질의 수를 제한한다. 결과는 `chunk_id`로 중복 제거한다. 병합은 RRF 같은 순위 기반 방법을 첫 안으로 한다.
- 비교 시 **고정**: 문서, 청킹, 임베딩, 평가셋, 최종 K, 답변 모델과 프롬프트.
- 후보 예산(Multi Query의 전체 후보 수, 결합 시 Rerank 입력 후보 수)을 명시한다.
- BM25·Hybrid는 이번 MVP의 추가 목표가 아니다.
- 개선이 없거나 성능이 떨어진 결과도 보고한다.

## 7. 샘플 데이터 사용 규칙

- 샘플 전달 전에는 가짜 데이터를 **인터페이스 연결 확인에만** 쓴다.
- 샘플 청크는 A가 실제 PDF의 대표 조문을 확인해 만든다. 샘플에 골든셋 정답 조문이 없으면 해당 질문은 샘플로 평가할 수 없다.
- **샘플 결과를 전체 문서 검색 성능으로 보고하지 않는다.** 전체 청크가 전달되면 컬렉션만 교체해 같은 실험을 회귀 평가한다.
- 샘플 결과는 `evaluation/results/sample/`에 분리하고 `chunks_version`을 `sample`로 기록한다.

## 8. 결과 기록

```
evaluation/results/{실험명}_{날짜}.json          # 전체 청크 결과
evaluation/results/sample/{실험명}_{날짜}.json   # 샘플 결과
```

```json
{
  "experiment": "baseline | rerank | multi_query | combined",
  "config": {
    "data_version": "v1", "chunking": "조 단위, 긴 조문은 항 단위",
    "embedding_model": "...", "chat_model": "...", "rerank": "전용 API | LLM 기반",
    "candidate_k": 20, "final_k": 5, "num_expanded_queries": 3,
    "golden_set_version": "v1", "split": "tune | final", "code_version": "..."
  },
  "metrics": {
    "overall": {"hit@5": 0.0, "recall@5": 0.0, "mrr": 0.0, "recall@candidate_k": 0.0},
    "by_type": {"definition": {}, "paraphrase": {}}
  },
  "cost": {"llm_calls_per_query": 0, "tokens_per_query": 0, "latency_sec": {"retrieve": 0, "rerank": 0, "generate": 0, "total": 0}},
  "per_question": [
    {"id": "q01", "retrieved_articles": ["제2조", "제33조"], "hit@5": true, "rank": 1}
  ]
}
```

`per_question`을 남기면 지표 정의를 바꿔도 재계산할 수 있고, 질문별로 어느 방식이 이겼는지 분석할 수 있다.

## 9. 운영 규칙

1. 청킹·구조 규칙이 바뀌면 데이터 버전을 올리고 **baseline부터 다시** 측정한다.
2. 골든셋을 점수를 올리려고 바꾸지 않는다. final 문항은 튜닝에 쓰지 않는다.
3. 설정을 바꾸면 결과 파일의 `config`에 반영한다.
4. 지표 계산은 C의 평가 실행기 하나만 쓴다. 각자 따로 구현하지 않는다.
5. API 키는 코드·노트북·보고서에 저장하지 않는다.

## 10. 합의 체크리스트

- [ ] MonoRouter: 모델 ID, 채팅·임베딩·Rerank 지원 여부, 비용 한도
- [ ] 임베딩 모델·차원 (잠정: `text-embedding-3-small`, 1536)
- [ ] 청크 필수 필드와 `article` 표기 형식 (`제N조`, `제N조의M`) — A와 합의
- [ ] 전체 청크 파일명과 컬렉션 이름
- [ ] `candidate_k`=20, 최종 K=5, 확장 질의 ≤3
- [ ] 답변 모델·프롬프트, LLM 채점 모델 (답변 모델과 다른 모델) — 채점 척도(0/1)와 항목 기준은 위 표로 제안, C와 합의
- [ ] 골든셋 정답 조문·항의 원문 대조 (팀 전원)
- [ ] 문서 버전·시행일의 본문 확인
