"""Synthetic MVP CLI tests; no external model or database."""

import json
from types import SimpleNamespace
from uuid import UUID

import pytest

from scripts.ingest import main


class FakeEmbeddings:
    model = "synthetic-model"

    def __init__(self):
        self.inputs = []

    def embed_documents(self, texts):
        self.inputs.append(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeQdrant:
    def __init__(self, exists=False):
        self.exists = exists
        self.calls = []
        self.points = []

    def collection_exists(self, collection_name):
        self.calls.append(("exists", collection_name))
        return self.exists

    def create_collection(self, collection_name, vectors_config):
        self.calls.append(("create", collection_name, vectors_config))
        return True

    def upsert(self, collection_name, points, wait):
        self.calls.append(("upsert", collection_name, wait))
        self.points = points
        return SimpleNamespace(status="completed")

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        self.calls.append(("retrieve", collection_name, ids, with_payload, with_vectors))
        return [SimpleNamespace(id=p.id, payload=p.payload) for p in reversed(self.points)]


@pytest.fixture
def chunk_file(tmp_path):
    chunks = [
        {
            "schema_version": "data-mvp-v1", "chunk_id": f"synthetic-{i}",
            "document_id": "synthetic", "document_version": "synthetic-v1",
            "pdf_sha256": "a" * 64, "source_uri": "synthetic://fixture",
            "article": f"제{i}조", "article_title": "합성 테스트",
            "content": f"합성 본문 {i}", "page_start": i, "page_end": i,
            "source_spans": [{"page": i, "start": 0, "end": 7}],
            "normalization_version": "verbatim-v1", "chunking_version": "article-v1",
        }
        for i in [1, 2]
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    return path, chunks


def test_write_creates_new_collection_and_verifies_all_payloads(chunk_file, capsys):
    path, chunks = chunk_file
    embeddings, client = FakeEmbeddings(), FakeQdrant()
    assert main([str(path), "--write", "--collection", "data_ingestion_synthetic_v1"],
                embedding_model=embeddings, qdrant_client=client) == 0
    assert embeddings.inputs == [[c["content"] for c in chunks]]
    assert [c[0] for c in client.calls] == ["exists", "create", "upsert", "retrieve"]
    config = client.calls[1][2]
    assert config.size == 2 and config.distance == "Cosine"
    assert [p.payload for p in client.points] == chunks
    assert [p.vector for p in client.points] == [[0.1, 0.2], [0.1, 0.2]]
    assert len({UUID(p.id) for p in client.points}) == len(chunks)
    assert client.calls[2][2] is True
    assert client.calls[3][2] == [p.id for p in client.points]
    assert client.calls[3][3:] == (True, False)
    assert json.loads(capsys.readouterr().out)["verified_count"] == 2


def test_default_dry_run_never_initializes_or_calls_services(chunk_file, monkeypatch, capsys):
    import sys

    def forbidden():
        pytest.fail("dry-run must not initialize services")

    monkeypatch.setitem(sys.modules, "common.ai_model", SimpleNamespace(get_embedding_model=forbidden))
    monkeypatch.setitem(sys.modules, "common.qdrant", SimpleNamespace(get_qdrant_client=forbidden))
    path, _ = chunk_file
    assert main([str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dry-run" and result["chunk_count"] == 2


def test_existing_collection_stops_before_embedding_or_mutation(chunk_file):
    path, _ = chunk_file
    embeddings, client = FakeEmbeddings(), FakeQdrant(exists=True)
    assert main([str(path), "--write", "--collection", "data_ingestion_synthetic_v1"],
                embedding_model=embeddings, qdrant_client=client) == 1
    assert embeddings.inputs == []
    assert client.calls == [("exists", "data_ingestion_synthetic_v1")]
