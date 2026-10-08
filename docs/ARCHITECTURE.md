# 아키텍처

## 현재 구조

2026-10-08 확인 기준 `app/main.py`에는 FastAPI 앱과 GET `/`가 있고 POST `/ask`는 없다. 기본 응답과 제품 QA 구현을 구분한다.

`app/`은 API, `common/config.py`는 환경변수 설정, `common/ai_model.py`는 ChatOpenAI/OpenAIEmbeddings 생성, `common/qdrant.py`는 QdrantClient 생성, `notebook/`은 실험, `docker-qdrant/`는 Compose와 기존 협업 계획이다. 루트 `pyproject.toml`과 `uv.lock`을 공유한다. 환경을 노트북별로 만들지 않는다.

`AGENTS.md`는 Codex 공통 지침, `.agents/skills/`는 계획과 리뷰, `scripts/execute.py`는 step 실행과 AC 검증, `scripts/ac_guard.py`는 AC 명령 검사, `phases/`는 작업 상태, `docs/`는 하네스 설계 문서다.

## 목표 데이터 흐름과 계약

PDF → 구조와 출처를 보존한 Chunk → 임베딩 → Qdrant → Dense 후보 → 선택적 질의 확장/병합·Rerank → Context → 답변과 실제 사용 출처.
Chunk와 검색 결과 계약의 상세 제안은 `docker-qdrant/docs/project-plan.md`에서 확인한다. chunk_id, 문서 버전, 조·항·페이지를 끝까지 보존한다. 검색 담당자는 공통 Dense 인터페이스를 사용하고 별도 API 서버를 만들지 않는다. 전체 적재와 컬렉션 변경은 데이터 담당자가 관리한다.

모델 제공자는 현재 OpenAI 호환 MonoRouter다. 환경 설정의 API_KEY/BASE_URL/MODEL/EMBEDDING_MODEL/QDRANT_URL 변수와 실제 로딩 구현을 확인하고 사용한다. 지원 기능·임베딩 차원은 호출로 확인하기 전 확정하지 않는다. 저장소에는 키 값을 기록하지 않는다.

## 상세 기획의 계약 제안

[사용자 저니와 구현 계획](PRODUCT_PLAN.md)의 문서·청크·검색·답변 계약을 기준으로 `common/contracts.py`부터 구현하는 순서를 제안한다. 결과 상태, 주장별 인용, 실제 발췌와 확인 한계를 구분한다. 인용 구조 검증과 의미적 충실성 평가를 별도로 수행한다. 계약 확정이나 구현 완료를 뜻하지 않는다.

웹 화면은 기존 MVP 범위 밖이다. 확장 시 같은 FastAPI 서비스에서 작은 정적 화면을 제공하는 방안을 제안한다. 일반 사용자에게 검색 모드는 노출하지 않으며 초기 재질문은 독립 요청이다. 화면·원본 PDF 제공 방법은 범위 확정 뒤 결정한다.

## 테스트 배치

제품 단위·회귀 테스트는 `tests/`에 두고 외부 호출은 fake/mock으로 대체한다. 구현 step에서 필요에 맞게 테스트 도구를 추가하고 uv.lock을 갱신한다. Qdrant·모델 통합 테스트는 별도로 명시하고 실제 실행 여부를 보고한다. 하네스 자체 테스트는 `scripts/execute_selftest.py`이며 모델을 mock하고 임시 Git 저장소와 실제 bash AC로 재시도·재개·무결성을 검사한다.

## 에러 처리·보안

필수: 빈 질문 처리, 결과 없음과 모델 실패를 구분, 확인되지 않은 근거로 답변하지 않기, 출처 메타데이터 보존, 비밀값 출력 금지, 공유 컬렉션 보존. AC는 로컬에서 반복 가능한 검증만 사용한다. Codex step은 workspace-write 샌드박스로 실행하며 Claude 훅을 설치하지 않는다. 실행기 검증과 위험 명령 검사는 Codex의 모든 도구 사용을 통제하는 보안 경계가 아니다.
