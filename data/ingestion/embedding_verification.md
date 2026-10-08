# Step 2 임베딩 연결 검증 (2026-10-08)

- `common.ingestion.embed_chunks()`는 기존 `get_embedding_model()` 또는 주입된 `embed_documents()`를 소비한다. 공유 설정과 모델 구현은 변경하지 않았다.
- 입력 `content` 순서와 청크·출처를 보존하며 벡터 개수, 양수·동일 차원, 유한 숫자를 확인한다. 검증된 결과의 `record`에 provider/model/관찰 dimensions/chunk_count를 반환한다. 모델 오류 본문은 노출하지 않는다.
- 구현 전 새 테스트가 `EmbeddingError` 부재로 실패한 것을 확인했다(수집 오류, 종료 2). 구현 후 임베딩 fake 테스트 10건, 기존 PDF 추출·청킹 회귀 테스트 6건이 통과했다. 기존 테스트와 기대값은 변경하지 않았다.
- AC 직접 실행: `uv run --with pytest python -m pytest tests/test_ingestion_embedding.py -q` 종료 0, `uv run python -m compileall -q common/ingestion.py` 종료 0.
- 기존 실제 PDF 산출물 `chunks.jsonl`의 46개 청크를 fake로 연결해 전체 입력 순서, 벡터 위치, 출처 메타데이터 및 입력 불변을 확인했다. 이는 실제 임베딩 품질이나 전체 조문 의미 검증이 아니다.

해당 로컬 연결 검증 기록:

```json
{"provider":"fake","model":"synthetic-order-check","dimensions":2,"chunk_count":46}
```

위 2차원은 fake의 관찰값이며 실제 제공자의 차원이 아니다. 외부 호출·Qdrant 적재는 하지 않았다. 실제 모델 지원과 차원은 앞선 HTTP 429로 여전히 미확인이다. 사용자 변경이 있던 `common/config.py`와 `notebook/qdrant_test.ipynb`의 SHA-256은 scope.json의 기준값과 동일했다. 커밋·푸시는 수행하지 않았다.
