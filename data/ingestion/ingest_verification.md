# Step 3 적재 CLI 검증

- 목적: `scripts/ingest.py`의 기본 dry-run으로 청크 JSONL을 확인하고, 명시적 `--write --collection data_ingestion_<version>`에서만 새 컬렉션을 생성·적재·조회 검증한다. 이 prefix는 로컬 MVP 규칙이며 팀이 예약한 이름은 아니다.
- 기존 컬렉션이면 임베딩·생성·upsert 전에 중단한다. 벡터 수·차원·유한값 검증은 Step 2 함수를 재사용한다. chunk_id를 고정 namespace의 UUID5로 매핑하며 원래 flat payload와 source_spans를 그대로 보존한다. 적재 후 모든 예상 ID와 payload가 일치해야 성공한다.
- 생성 이후 실패한 컬렉션은 그대로 남긴다. 재실행은 기존 컬렉션 쓰기를 거부하며 삭제·재개·alias 관리는 구현하지 않는다.
- TDD: 구현 전 핵심 테스트 실행은 `scripts.ingest` 부재로 수집 실패(exit 2). 구현 후 정상 적재·dry-run 외부 초기화 없음·기존 컬렉션 보호 3건 통과. 기존 추출·청킹·임베딩 회귀 16건도 통과했으며 기존 테스트는 변경하지 않았다.
- AC 직접 실행: `uv run --with pytest python -m pytest tests/test_ingest_cli.py -q`, `uv run python scripts/ingest.py --help`, `uv run python -m compileall -q common/ingestion.py common/chunking.py scripts/ingest.py` 모두 exit 0.
- 실제 산출물: `data/ingestion/chunks.jsonl` 46건 dry-run 통과. 같은 46건을 주입 fake 임베딩·Qdrant에 적재하여 내용 순서와 모든 payload·ID 조회 일치를 확인했다. 제17조의2는 7페이지, 제22조의2는 9~10페이지 정보를 유지했다. fake 데이터베이스만 사용했으며 실제 컬렉션은 생성하지 않았다.
- 한계: 전체 조문의 의미·출처 품질을 새로 대조하지 않았다. 실제 모델 임베딩은 이전 HTTP 429로 미확인 상태이며 실제 Qdrant 통합·DB 쓰기·팀 컬렉션 예약도 미수행이다. 로컬 통과를 외부 인계 완료로 간주하지 않는다.
- 커밋·푸시 없음. 공유 설정·의존성·Notebook·계약·문서·하네스는 수정하지 않았다. Step 3 상태만 completed로 기록하며 completed_at 확정은 실행기의 AC 재검증에 맡긴다.
