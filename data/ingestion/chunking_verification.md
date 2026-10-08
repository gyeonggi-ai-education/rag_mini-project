# Step 1 조문 청킹 검증 — 2026-10-08

- `common/chunking.py`의 `chunk_articles(document, running_header=...)`는
  줄 시작의 `제N조(제목)` 및 `제N조의M(제목)`을 경계로 조문 전체를 유지한다.
  본문 내 조문 참조는 경계로 처리하지 않으며 페이지를 넘긴 조문을 연결한다.
- 조문 번호·제목·본문, 문서 ID/버전·해시·출처, 물리 PDF 페이지 범위와
  페이지별 source spans를 보존한다. spans는 `pages.jsonl` 원형 텍스트의
  Unicode code point 인덱스(start 포함, end 제외)다. 각 span의 문자열을
  개행 하나로 연결하면 content와 정확히 일치한다. 페이지 번호는 인쇄
  페이지를 추정한 값이 아니다.
- ID는 본문·문서 버전·조문·출처와 규칙 버전을 포함하는 canonical JSON의
  SHA-256이다. 같은 입력은 같은 ID를 생성한다. 잠정 내부 schema
  `data-mvp-v1`, 정규화 `verbatim-v1`, 청킹 `whole-article-v1`을 사용한다.
  팀 공통 계약이나 `common/contracts.py`는 변경하지 않았다.
- 실제 원본과 버전은 Step 0과 동일하다. `source_manifest.json`에 원본
  해시·로컬 URI·문서 ID `law-21311`·원문 버전 표기·15페이지를 기록했다.
  로컬 URI는 공유 URL이 아니다. 원본 PDF를 변경하지 않았다.
- 산출물: `pages.jsonl`에 원형 페이지 15건, `chunks.jsonl`에 전체 PDF에서
  발견한 조문 46건, `samples.jsonl`에 제17조의2와 제22조의2 2건.
  11개 조문은 여러 페이지에 걸친다. 조문 크기 분할은 하지 않았다.
- 실제 샘플 추출 텍스트 대조: 제17조의2는 PDF 7페이지, 제목
  `인공지능제품 및 인공지능서비스 이용비용의 지원`, ①·②와 본조신설 표기를
  보존했다. 제22조의2는 PDF 9~10페이지, 제목
  `인공지능연구소의 설립 및 지원 등`, ①~⑫ 및 본조신설 표기를 보존했다.
  다음 제22조의3은 별도 청크다. 이는 PDF 추출 텍스트 기준 대조이며
  렌더 이미지 대조나 법률 해석 검증은 아니다.
- 제외 범위: 첫 조문 이전 표지·버전 안내, 독립 장·절 제목, 호출자가
  확인해 전달한 반복 문서명 머리말과 `법제처 N 국가법령정보센터` 꼬리말.
  제거 전 원형 텍스트는 pages.jsonl에 보존했다. 본문·항·호·조건·예외의
  문자열은 정규화하지 않는다. 이 파일에 부칙 표기가 발견되지 않았다는
  이전 확인을 넘어서 부칙이 법령 자체에 없다고 판단하지 않는다.
- TDD: 구현 전 새 테스트는 `common.chunking` 부재로 수집 실패(exit 2).
  구현 후 같은 테스트 및 기존 추출 테스트 6개 통과(exit 0).
  조문 경계와 본문 참조 구분, 다중 페이지 출처, 머리말·꼬리말 제외,
  결정적 ID 및 버전·본문 변경 시 ID 변경을 synthetic 입력으로 검증했다.
- AC: `uv run --with pytest python -m pytest tests/test_pdf_extraction.py tests/test_chunking.py -q`
  및 `uv run python -m compileall -q common/ingestion.py common/chunking.py` 통과.
  `python3 scripts/execute.py data-ingestion --check`도 통과했다.
- 저장된 산출물을 다시 읽어 동일 청크 재생성, 46개 ID 중복 없음,
  모든 span의 페이지 범위·문자 범위·content 재구성 일치를 확인했다.
  기존 사용자 config와 Notebook의 scope 기준 SHA-256은 보존됐다.
- 한계: 전체 조문에 대한 사람의 원문·의미 대조, 범용 부칙 파싱,
  모델·임베딩·Qdrant 연동은 미검증이다. 외부 모델·DB 호출, 공유
  컬렉션 변경, 커밋·푸시는 수행하지 않았다. 다음 step은 실행하지 않았다.
