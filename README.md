# AI 기본법 근거 기반 QA (RAG Mini Project)

「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」(법률 제21311호)을 잘 모르는 사용자가 질문하면, 법률 원문에서 관련 조문을 검색해 **쉬운 답변과 출처 조문**을 돌려주는 RAG 시스템이다.
Qdrant 벡터 검색 위에 **Rerank**와 **Multi Query** 검색 개선 기법을 얹고, 같은 골든셋으로 비교 평가했다.

## 주요 기능

- `POST /ask` — 질문 → 검색 → 답변 + 실제 사용한 근거 조문
- 검색 모드 4종: `baseline`(Dense) / `rerank` / `multi_query` / `combined`
- 근거가 부족하면 "제공된 법률 문서에서는 해당 내용을 확인하기 어렵습니다."로 답변
- 웹 화면: 채팅(`/chat`), 컬렉션 뷰어(`/viewer`), 평가 결과 보기(`/eval`)
- 동일 골든셋 기반 평가 실행기 (Hit@K, Recall@K, MRR, 지연, 토큰)

## 아키텍처

```text
질문
 ├─ baseline     : Dense 검색(top-K)
 ├─ rerank       : Dense 후보 20개 → LLM Rerank → 최종 5개
 ├─ multi_query  : 질의 확장(최대 3개) → 질의별 Dense → RRF 병합 → 최종 5개
 └─ combined     : multi_query 병합 후보 → LLM Rerank → 최종 5개   (서버 기본값)
                         ↓
              근거 Context 구성 → LLM 답변 생성 → {answer, sources}
```

- 후보 수 `CANDIDATE_K=20`, 최종 `FINAL_K=5` ([common/config.py](common/config.py))
- Rerank는 전용 API가 아니라 **LLM 기반 재정렬**(후보별 0~10점)이다. 지연·토큰 비용이 늘어난다.
- 청킹은 조문 단위(`whole-article-v1`), 기본 컬렉션은 `data_ingestion_law21311_v1`(46청크)

## 디렉터리 구조

```text
app/          FastAPI 앱 (main.py: /ask, /chat · viewer.py: 컬렉션 뷰어 · eval_api.py: 평가 결과 API · static/)
common/       설정(config.py), LLM·임베딩 모델(ai_model.py), Qdrant 클라이언트(qdrant.py)
retrieval/    dense.py, rerank.py, multi_query.py, pipeline.py(모드별 retrieve)
generation/   answer.py (Context 구성, 답변·출처 생성)
evaluation/   골든셋, 평가 실행기(run_eval.py), 지표, 결과(results/), 평가 규칙(EVAL_PROTOCOL.md)
scripts/      eval_retrieval.py (baseline/rerank 평가)
notebook/     데이터 로딩·Qdrant 실험 노트북
data/raw/     원문 PDF (ai_basic_act.pdf)
docker-qdrant/ Qdrant 실행용 docker-compose
```

## 실행 방법

사전 준비: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker

```bash
# 1. 의존성 설치 (.venv 생성, common/app이 editable 패키지로 설치됨)
uv sync

# 2. Qdrant 실행
cd docker-qdrant && docker compose up -d && cd ..
#    대시보드: http://localhost:6333/dashboard

# 3. 환경변수 (.env, Git에 올리지 않는다)
#    LLM_API_KEY=...
#    LLM_BASE_URL=...
#    선택: LLM_MODEL, EMBEDDING_MODEL, QDRANT_URL, QDRANT_COLLECTION, RETRIEVAL_MODE

# 4. API 서버 실행
uv run uvicorn app.main:app --reload
```

| 주소 | 설명 |
|---|---|
| http://localhost:8000/chat | 채팅 화면 (`/`는 여기로 이동) |
| http://localhost:8000/docs | API 문서 (Swagger) |
| http://localhost:8000/viewer | Qdrant 컬렉션·청크 뷰어 |
| http://localhost:8000/eval | 평가 결과 보기 |

노트북은 `uv run jupyter lab`으로 실행하고, VS Code에서는 `.venv`의 Python을 커널로 선택한다.

### 환경변수

| 이름 | 기본값 | 설명 |
|---|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL` | - | 답변·질의 확장·Rerank용 LLM 엔드포인트 (OpenAI 호환) |
| `LLM_MODEL` | `gpt-5.4-mini` | 사용할 LLM |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 문서·질문 임베딩 (둘 다 같은 설정 사용) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 주소 |
| `QDRANT_COLLECTION` | `data_ingestion_law21311_v1` | 검색 대상 컬렉션 |
| `RETRIEVAL_MODE` | `combined` | `/ask` 기본 검색 모드 |
| `LLM_MAX_RETRIES` | `1` | 클라이언트 재시도 횟수 (호출 제한이 있는 키를 공유하므로 낮게 유지) |

## API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "고영향 인공지능이란 무엇인가요?"}'
```

```json
{
  "answer": "검색 근거에 기반한 설명",
  "sources": [{"article": "제○조", "content": "실제로 사용한 법률 내용"}],
  "mode": "combined"
}
```

- 요청에 `"mode"`를 추가하면 검색 모드를 바꿀 수 있다 (`baseline`, `rerank`, `multi_query`, `combined`). 생략하면 서버 기본 모드를 쓴다.
- `sources`는 검색 결과 전체가 아니라 **답변에 실제로 사용한 근거**만 담는다.
- 빈 질문은 400, 지원하지 않는 mode는 400, 모델·검색 호출 실패는 502를 반환한다.

## 평가

골든셋(10문항, `evaluation/golden_set.jsonl`)으로 네 모드를 같은 조건(문서·청킹·임베딩·K=5)에서 비교한다. 자세한 규칙은 [evaluation/EVAL_PROTOCOL.md](evaluation/EVAL_PROTOCOL.md)를 따른다.

```bash
# 검색 지표 (결과: evaluation/results/)
uv run python -m evaluation.run_eval --k 5

# 답변 생성·규칙 기반 채점까지
uv run python -m evaluation.run_eval --k 5 --answer shared

# 빠른 확인 / 캐시 끄기
uv run python -m evaluation.run_eval --k 5 --limit 3 --no-cache
```

### 결과 (골든셋 10문항, K=5, 실행 `20261008_155314`)

| 모드 | Hit@5 | Recall@5 | MRR | 평균 지연(초) | 평균 토큰 |
|---|---|---|---|---|---|
| baseline | 0.90 | 0.85 | 0.85 | 0.003 | 0 |
| rerank | 1.00 | 0.95 | 0.95 | 3.18 | 7,484 |
| multi_query | 1.00 | 0.90 | 1.00 | 0.015 | 0 |
| combined | 1.00 | 0.95 | 0.95 | 3.59 | 7,420 |

> 10문항의 소규모 탐색 결과이며 일반화할 수 없다. multi_query의 지연은 질의 확장·임베딩 캐시가 있는 상태의 값이다(캐시 없는 첫 실행은 확장 LLM 호출로 수 초 소요). 결과 원본은 [evaluation/results/](evaluation/results/)에서 확인할 수 있다.

## 협업 구조

| 담당 | 역할 |
|---|---|
| A | 데이터·청킹, 임베딩·Qdrant 적재 |
| B | Dense 검색, Rerank, FastAPI `/ask`, 답변·출처 생성 |
| C | Multi Query, 평가 실행기·지표, 실험 기록 |

## 주의사항

- `.env`의 API 키는 저장소·노트북·보고서에 기록하지 않는다.
- 공유 컬렉션을 덮어쓰지 않는다. 재청킹 실험은 새 컬렉션·데이터 버전으로 분리한다.
- LLM 호출에 분당 제한이 있어 평가 실행기는 캐시(`evaluation/.cache`)와 429 대기·재시도를 사용한다.
