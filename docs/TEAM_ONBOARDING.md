# 팀원 온보딩 — 로컬 Qdrant 복제

이 묶음은 AI 기본법 PDF의 **46개 청크와 실제 임베딩 벡터**를 포함한다. Qdrant 1.19.2, 1536차원 Cosine, 컬렉션명은 `data_ingestion_law21311_v1`이다. 복구할 때 임베딩 API를 다시 호출하지 않는다.

## 1. ZIP을 풀고 환경 실행

Python 3.12 이상, uv, Docker가 필요하다. 압축을 푼 프로젝트 폴더에서:

```bash
uv sync --locked
docker compose -f docker-qdrant/docker-compose.replica.yaml up -d
```

이미 6333 포트를 사용하는 Qdrant 1.19.2가 있으면 그것을 사용하고 `up`은 생략한다. 기존 컨테이너나 볼륨을 삭제하지 않는다.

## 2. 스냅샷 복구

아래 코드는 해시와 대상 컬렉션 부재를 확인한다. 이미 같은 컬렉션이 있으면 중단하며, 삭제하지 말고 `target`을 다른 새 이름으로 바꾼다. 같은 이름으로 복구와 적재를 동시에 실행하지 않는다.

```bash
uv run python - <<'PY'
import hashlib, json
from pathlib import Path
import httpx
from qdrant_client import QdrantClient

base = "http://127.0.0.1:6333"
target = "data_ingestion_law21311_v1"
folder = Path("data/qdrant")
manifest = json.loads((folder / "manifest.json").read_text())
snapshot = folder / manifest["snapshot_file"]
with snapshot.open("rb") as stream:
    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
assert checksum == manifest["snapshot_sha256"], "스냅샷 해시 불일치"
client = QdrantClient(url=base)
try:
    if client.collection_exists(target):
        raise SystemExit("이미 있는 컬렉션입니다. 다른 새 이름을 선택해 주세요.")
    with httpx.Client(timeout=60) as http:
        version = http.get(base + "/")
        version.raise_for_status()
        assert version.json()["version"] == manifest["qdrant_version"], "Qdrant 1.19.2를 사용해 주세요."
        with snapshot.open("rb") as stream:
            result = http.post(
                base + "/collections/" + target + "/snapshots/upload",
                params={"priority": "snapshot", "wait": "true", "checksum": checksum},
                files={"snapshot": (snapshot.name, stream, "application/octet-stream")},
            )
            result.raise_for_status()
            assert result.json().get("result") is True
    info = client.get_collection(target)
    assert info.points_count == manifest["points_count"]
    print("복구 완료:", target, info.points_count, "개 청크")
finally:
    client.close()
PY
```

## 3. Qdrant에서 청크 보기

브라우저에서 **http://localhost:6333/dashboard**를 열고 `data_ingestion_law21311_v1`을 선택한다. Points에서 payload의 `content`, `article`, `page_start`, `page_end`, `source_spans`를 보면 청크 본문과 출처를 확인할 수 있다.

코드에서는 공통 `get_qdrant_client()`를 사용하고 컬렉션명을 명시한다. 개인 `.env`의 `QDRANT_URL`도 로컬 주소로 맞춘다. `.env`와 키는 압축에 포함하지 않았으며, 질문 임베딩·답변 생성에 필요한 제공자 키는 각자 설정한다. 질문 검색에는 적재와 같은 `text-embedding-3-small`을 사용한다.

## 들어 있는 데이터와 코드

- `data/qdrant/`: 46개 청크의 스냅샷, 해시·차원 manifest, 실제 적재 기록.
- `data/ingestion/source.pdf`: 원본 15페이지 PDF.
- `data/ingestion/chunks.jsonl`, `pages.jsonl`, `source_manifest.json`: 청킹·추출 결과와 원본 해시·출처.
- `common/`, `app/`, 적재·평가 CLI, `pyproject.toml`, `uv.lock`: 같은 실행 환경과 공통 모듈.

원본 출처 URI와 조·항·페이지를 유지했다. 기록의 개인 파일 URI 대신 제공된 `source.pdf`를 로컬에서 열 수 있다. 원본 해시가 같은지 확인하고 출처를 임의로 만들지 않는다. 청킹 변경은 새 버전 컬렉션에서 검증한다. 개인 복제본은 자동 동기화되지 않으므로 새 데이터를 공유할 때 새 스냅샷으로 전달한다.

현재 기본 `/ask`는 서비스 설정 전 503이며 기존 JSONL payload와 API Chunk 계약 연결은 별도 작업이다. 이번 확인은 **46개 실제 임베딩 적재와 전체 payload 조회 일치**까지다. 법률 답변 정확도나 이번 46개 스냅샷의 별도 복구 테스트는 추가 수행하지 않았다.

복구 API는 기존 컬렉션을 덮어쓸 수 있어 위 사전 검사를 유지한다. 공유 컬렉션 삭제·재생성, `docker compose down -v`는 하지 않는다. [Qdrant 스냅샷 문서](https://qdrant.tech/documentation/operations/snapshots/), [업로드 복구 API](https://api.qdrant.tech/api-reference/snapshots/recover-from-uploaded-snapshot)
