# C 작업 체크리스트 (A 공유 → C 작업 → B 공유)

A가 공유한 Qdrant 스냅샷(`rag-team-share/`)으로 실제 데이터를 평가하고, 그 결과로 `HANDOFF_B.md`의 ⚠️ 부분을 채워 B에게 최종 공유한다.
A는 서버 주소가 아니라 **스냅샷 파일**로 공유했다 (2026-10-08). 로컬 복구 완료.
호출 수는 실제 골든셋 10문항 기준 계산값이다.

## 1. A 공유 전 (지금)

- [x] Multi Query 모듈, 평가 실행기, 결과 조회·리포트·API 라우터, 노트북
- [x] 호출 절감: 확장·임베딩 묶음, 캐시, 답변 모드 선택, 규칙 채점, Rerank 캐시
- [x] B 전달 문서 초안 (`HANDOFF_B.md`, 더미 기준)
- [ ] 더미 데이터로 실제 API 테스트 (검색만 2회 → 재실행 0회 확인, 확장 프롬프트 v2 효과 확인)
- [x] 컬렉션 점검 스크립트 `evaluation/check_collection.py` (Qdrant만 조회, API 0회)
- [ ] (선택) `evaluation/temp_ingest.py` 안전장치: `QDRANT_URL`이 localhost가 아니면 실행 거부
- [ ] 실제 골든셋 답변 불가 2문항 추가 (팀과 원문 확인)

## 2. A에게 받은 정보 (`rag-team-share/`, 2026-10-08)

- [x] 공유 방식: 스냅샷 파일 → 로컬 Qdrant 1.19.2에 복구 (덮어쓰기 방지 검사 포함, API 0회)
- [x] 컬렉션: `data_ingestion_law21311_v1` (C 평가 기본값으로 설정, `QDRANT_COLLECTION`으로 변경 가능)
- [x] 임베딩: `text-embedding-3-small`, 1536차원, 코사인 (질문 임베딩과 같음)
- [x] 데이터 버전: `[시행 2026. 7. 21.] [법률 제21311호, 2026. 1. 20., 일부개정]`, 청킹 `whole-article-v1` (조 단위 46개, 부칙 없음)
- [x] payload: `chunk_id`(64자리 해시), `article`, `article_title`, `content`, `page_start`·`page_end`, `source_spans` 등. `paragraph`·`is_supplementary` 없음
- [x] 점검: 필수 필드 있음, 골든셋 정답 조문 11/11, 공유 JSONL과 payload 일치 46개
- [ ] 재청킹 시 새 컬렉션 이름을 쓰겠다는 합의 (A 문서에 "청킹 변경은 새 버전 컬렉션"으로 적혀 있음, 확인만)

## 3. A 공유 직후 (순서대로)

| 순서 | 할 일 | 명령·위치 | MonoRouter 호출 | 상태 |
|---|---|---|---|---|
| 1 | 스냅샷 복구 | `rag-team-share/docs/TEAM_ONBOARDING.md` 2장 | 0 | 완료 |
| 2 | 컬렉션 점검 | `check_collection --chunks rag-team-share/data/ingestion/chunks.jsonl` | 0 | 완료 (통과) |
| 3 | C 코드 맞추기: 기본 컬렉션, 해시 chunk_id를 조 번호로 표시 | `chunk_source.py`, `results_view.py`, `report.py`, `trace.py` | 0 | 완료 |
| 4 | 빠른 확인 | `run_eval --k 5 --limit 3 --data-version law21311-whole-article-v1` | 2 | |
| 5 | 검색 평가 전체 | `run_eval --k 5 --data-version law21311-whole-article-v1` | 0~1 (4에서 캐시) | |
| 6 | 결과 확인, 실패 질문 메모 | `results_view --detail`, `report` | 0 | |
| 7 | (필요 시) 확장 프롬프트 조정 후 `--split tune`으로 재평가 | `retrieval/multi_query.py` 버전 올리기 | 2 | |

## 4. B 공유 전 갱신 (`HANDOFF_B.md`의 ⚠️ 부분)

- [ ] 3장 엔드포인트 예시: 실제 골든셋 문항 id, A의 chunk_id 형식
- [ ] 5장 청크 필드: A의 실제 payload 필드
- [ ] 6장 호출 수: 실측 `monorouter_requests`로 교체
- [ ] 7장 명령: 실제 컬렉션 이름, 데이터 버전
- [ ] 8장 실제 결과: 모드별 지표 표, 실측 호출 수, 대표 실패 사례
- [ ] 맨 위 "초안" 표시 제거

## 5. B에게 받을 것 (받으면 C가 연결)

- [ ] `REQUEST_TO_B.md` 공유 → B가 작성해서 돌려주면 C 코드를 미리 맞추고, 코드가 올라오면 코드로 최종 확인

- [ ] `retrieval/rerank.py` `rerank(question, candidates, final_k)` → rerank·combined 자동 실행
- [ ] `generation/answer.py` `generate_answer(question, chunks)` → `--answer shared`
- [ ] combined 흐름 확인 (병합 후 Rerank 1회, 후보 20, K=5) 또는 `retrieve(question, mode)` 공유 함수
- [ ] 최종 서비스 모드 → `--answer-modes baseline <최종 모드>`로 기본값 변경
- [ ] Rerank 방식 (한 번의 호출 / 로컬 모델) → 6장 호출 수 갱신

## 6. 최종 평가와 공유

- [ ] B의 Rerank·답변 함수 연결 후 최종 평가 1회 (약 38회, Rerank를 로컬 모델로 하면 약 18회)
  ```bash
  uv run python -m evaluation.run_eval --collection <이름> --k 5 --answer shared \
    --answer-modes baseline <최종 모드> --data-version <버전>
  ```
- [ ] `/ask` 시연·테스트와 같은 시간에 돌리지 않기 (분당 30회 한도 공유)
- [ ] 리포트에서 실패 사례 골라 보고서 C 담당 부분 작성 (지표, 실패 사례, 시도 전후)
- [ ] `HANDOFF_B.md` 최종본 공유, B가 `app/main.py`에 라우터 연결
- [ ] git: C 담당 파일만 커밋 (`evaluation/`, `retrieval/multi_query.py`, `tests/`, 노트북 `c_multi_query_evaluation.ipynb`)
