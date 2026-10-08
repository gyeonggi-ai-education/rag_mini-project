from types import SimpleNamespace
import sys

import pytest

from common.ingestion import EmbeddingError, embed_chunks


class FakeEmbeddings:
    model = "synthetic-model"

    def __init__(self, vectors):
        self.vectors = vectors
        self.inputs = []

    def embed_documents(self, texts):
        self.inputs.append(texts)
        return self.vectors


def test_content_order_mapping_and_profile_preserve_chunks():
    chunks = [
        {"chunk_id": "b", "content": "두 번째 조문", "page_start": 2},
        {"chunk_id": "a", "content": "첫 번째 조문", "page_start": 1},
    ]
    fake = FakeEmbeddings([[0.1, 0.2], [0.3, 0.4]])
    result = embed_chunks(chunks, embedding_model=fake, provider="fake")
    assert fake.inputs == [["두 번째 조문", "첫 번째 조문"]]
    assert result.chunks == tuple(chunks)
    assert result.vectors == ((0.1, 0.2), (0.3, 0.4))
    assert result.record == {
        "provider": "fake", "model": "synthetic-model",
        "dimensions": 2, "chunk_count": 2,
    }
    assert "vector" not in chunks[0]


def test_default_uses_existing_model_factory(monkeypatch):
    fake = FakeEmbeddings([[1, 2]])
    monkeypatch.setitem(
        sys.modules, "common.ai_model",
        SimpleNamespace(get_embedding_model=lambda: fake),
    )
    result = embed_chunks([{"content": "합성 본문"}], provider="fake")
    assert fake.inputs == [["합성 본문"]]
    assert result.record["model"] == fake.model


@pytest.mark.parametrize("vectors", [
    [], [[1, 2], [3, 4]], [[]], [[float("nan")]],
    [[float("inf")]], [["1"]], [[True]],
])
def test_invalid_vector_response_is_rejected(vectors):
    with pytest.raises(EmbeddingError):
        embed_chunks(
            [{"content": "합성 본문"}],
            embedding_model=FakeEmbeddings(vectors), provider="fake",
        )


def test_inconsistent_dimensions_are_rejected():
    with pytest.raises(EmbeddingError):
        embed_chunks(
            [{"content": "본문 A"}, {"content": "본문 B"}],
            embedding_model=FakeEmbeddings([[1, 2], [3]]), provider="fake",
        )
