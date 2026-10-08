"""Read-only Dense retrieval over one explicitly selected document version.

Pass the embedding record returned by ingestion and the same embedding adapter
(or the shared factory configuration). Provider/model/dimensions are checked;
the caller must also preserve provider endpoint and any custom adapter options.
No model name establishes provider support. Scores are internal cosine search
scores, never legal confidence. No proposed candidate count or K is fixed here.
"""

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from common.contracts import AskRequest, Chunk, SearchHit, SearchMode


class RetrievalError(RuntimeError):
    """Dependency or stored-data failure, with no raw provider details."""


def _positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"Expected positive integer {label}")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RetrievalError("Expected finite numeric retrieval data")
    try:
        converted = float(value)
    except (ValueError, OverflowError):
        raise RetrievalError("Expected finite numeric retrieval data") from None
    if not math.isfinite(converted):
        raise RetrievalError("Expected finite numeric retrieval data")
    return converted


class DenseRetriever:
    """Configured scope with retrieve(question, mode, k) -> list[SearchHit].

    embedding_record is the ingestion result, not guessed environment metadata.
    candidate_count is optional: absent it, query exactly k candidates. Duplicate
    candidates may yield fewer than k hits; no unbounded refill is performed.
    Only Dense is implemented; other valid contract modes fail explicitly.
    """

    def __init__(
        self, *, collection: str, document_id: str, document_version: str,
        embedding_record: Mapping, provider: str = "monorouter",
        candidate_count: int | None = None, embedding_model=None, qdrant_client=None,
    ):
        for value in (collection, document_id, document_version, provider):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Expected explicit collection, document, version and provider")
        if not isinstance(embedding_record, Mapping):
            raise ValueError("Expected ingestion embedding record")
        model = embedding_record.get("model")
        dimensions = embedding_record.get("dimensions")
        if embedding_record.get("provider") != provider:
            raise ValueError("Embedding provider differs from indexing record")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Expected indexed embedding model label")
        _positive_integer(dimensions, "dimensions")
        if candidate_count is not None:
            _positive_integer(candidate_count, "candidate_count")
        # Import factories only when needed; injected unit tests load no secrets.
        try:
            if embedding_model is None:
                from common.ai_model import get_embedding_model

                embedding_model = get_embedding_model()
        except Exception:
            raise RetrievalError("Embedding initialization failed") from None
        if getattr(embedding_model, "model", None) != model:
            raise ValueError("Embedding model differs from indexing record")
        try:
            if qdrant_client is None:
                from common.qdrant import get_qdrant_client

                qdrant_client = get_qdrant_client()
        except Exception:
            raise RetrievalError("Qdrant initialization failed") from None
        self.collection = collection
        self.document_id = document_id
        self.document_version = document_version
        self.dimensions = dimensions
        self.indexed_model = model
        self.provider = provider
        self.candidate_count = candidate_count
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client
        self.last_calls = {}

    def retrieve(self, question: str, mode: SearchMode, k: int) -> list[SearchHit]:
        self.last_calls = {}
        request = AskRequest(question=question, search_mode=mode)
        if request.search_mode != "dense":
            raise ValueError("Only Dense retrieval is implemented")
        _positive_integer(k, "k")
        limit = k if self.candidate_count is None else self.candidate_count
        if limit < k:
            raise ValueError("candidate_count must be at least k")
        if getattr(self.embedding_model, "model", None) != self.indexed_model:
            raise RetrievalError("Embedding model differs from indexing record")
        try:
            self.last_calls["collection"] = 1
            info = self.qdrant_client.get_collection(collection_name=self.collection)
            vectors = info.config.params.vectors
            # Ingestion creates an unnamed, cosine collection. Named vectors
            # require an explicit future contract rather than choosing one.
            if (getattr(vectors, "size", None) != self.dimensions
                    or getattr(vectors, "distance", None) != "Cosine"):
                raise RetrievalError("Collection vector settings differ from indexing")
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError("Qdrant collection check failed") from None
        try:
            self.last_calls["embedding"] = 1
            vector = self.embedding_model.embed_query(request.question)
        except Exception:
            raise RetrievalError("Query embedding failed") from None
        if (not isinstance(vector, Sequence) or isinstance(vector, (str, bytes))
                or len(vector) != self.dimensions):
            raise RetrievalError("Query vector dimensions differ from indexing")
        vector = [_number(value) for value in vector]
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=self.document_id)),
            FieldCondition(key="document_version", match=MatchValue(value=self.document_version)),
        ])
        try:
            self.last_calls["query"] = 1
            result = self.qdrant_client.query_points(
                collection_name=self.collection, query=vector, query_filter=query_filter,
                limit=limit, with_payload=True, with_vectors=False,
            )
            points = result.points
            if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
                raise RetrievalError("Invalid Qdrant search response")
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError("Qdrant search failed") from None
        by_id = {}
        for point in points:
            try:
                payload = point.payload
                if not isinstance(payload, Mapping):
                    raise RetrievalError("Invalid stored chunk payload")
                if (payload.get("document_id") != self.document_id
                        or payload.get("document_version") != self.document_version):
                    raise RetrievalError("Search result is outside requested document version")
                # Ingestion adds document metadata to the flat Chunk payload.
                # Project only contract fields, retaining exact text/provenance.
                chunk = Chunk.model_validate({
                    key: payload[key] for key in Chunk.model_fields if key in payload
                })
                score = _number(point.score)
            except RetrievalError:
                raise
            except Exception:
                raise RetrievalError("Invalid stored chunk or search score") from None
            previous = by_id.get(chunk.chunk_id)
            if previous is not None:
                if previous[0] != chunk:
                    raise RetrievalError("Duplicate chunk ID has conflicting content or provenance")
                score = max(score, previous[1])
            by_id[chunk.chunk_id] = (chunk, score)
        ordered = sorted(by_id.values(), key=lambda item: (-item[1], item[0].chunk_id))
        return [SearchHit(chunk=chunk, rank=rank, retrieval_score=score)
                for rank, (chunk, score) in enumerate(ordered[:k], start=1)]


class StrategyRetriever:
    """Explicit adapters over the same scoped Dense index, without API guesses.

    expand(question, maximum) returns a finite sequence of additional questions.
    rerank(question, chunks) returns one finite score per chunk, in input order.
    Adapters receive copies. No default Rerank provider is assumed. Multi Query
    merges by reciprocal rank (sum 1/(60+rank)), with deterministic ID tie breaks.
    candidate_count is per query; total_candidate_budget bounds all query slots.
    last_calls counts adapter requests, including failed attempts, not internal
    retries, HTTP traffic, or token usage. Instances are for sequential requests.
    """

    def __init__(self, dense: DenseRetriever, *, candidate_count: int,
                 max_expansions: int = 3, total_candidate_budget: int | None = None,
                 expand=None, rerank=None):
        _positive_integer(candidate_count, "candidate_count")
        if type(max_expansions) is not int or max_expansions < 0:
            raise ValueError("Expected nonnegative max_expansions")
        if total_candidate_budget is None:
            total_candidate_budget = candidate_count * (1 + max_expansions)
        _positive_integer(total_candidate_budget, "total_candidate_budget")
        if dense.candidate_count not in (None, candidate_count):
            raise ValueError("Dense and strategy candidate counts differ")
        self.dense = dense
        self.candidate_count = candidate_count
        self.max_expansions = max_expansions
        self.total_candidate_budget = total_candidate_budget
        self.expand = expand
        self.rerank = rerank
        self.last_calls = {}

    @property
    def document_id(self):
        return self.dense.document_id

    @property
    def document_version(self):
        return self.dense.document_version

    @property
    def collection(self):
        return self.dense.collection

    def _dense(self, question):
        try:
            return self.dense.retrieve(question, "dense", self.candidate_count)
        finally:
            for label, count in self.dense.last_calls.items():
                self.last_calls[label] = self.last_calls.get(label, 0) + count

    def retrieve(self, question: str, mode: SearchMode, k: int) -> list[SearchHit]:
        self.last_calls = {}
        request = AskRequest(question=question, search_mode=mode)
        _positive_integer(k, "k")
        if k > self.candidate_count:
            raise ValueError("candidate_count must be at least k")
        multi = request.search_mode in {"multi_query", "multi_query_rerank"}
        ranked = request.search_mode in {"rerank", "multi_query_rerank"}
        if multi and self.expand is None:
            raise ValueError("Query expansion adapter is not configured")
        if ranked and self.rerank is None:
            raise ValueError("Rerank adapter is not configured")
        slots = 1 + self.max_expansions if multi else 1
        if self.candidate_count * slots > self.total_candidate_budget:
            raise ValueError("Total candidate budget is too small")
        questions = [request.question]
        if multi and self.max_expansions:
            self.last_calls["expansion"] = 1
            try:
                expanded = self.expand(request.question, self.max_expansions)
                if not isinstance(expanded, Sequence) or isinstance(expanded, (str, bytes)):
                    raise ValueError("Expected finite query sequence")
                for value in expanded:
                    value = AskRequest(question=value).question
                    if value not in questions:
                        questions.append(value)
                    if len(questions) == 1 + self.max_expansions:
                        break
            except Exception:
                raise RetrievalError("Query expansion failed") from None
        by_id, fusion = {}, {}
        for query in questions:
            for hit in self._dense(query):
                cid = hit.chunk.chunk_id
                old = by_id.get(cid)
                if old is not None and old.chunk != hit.chunk:
                    raise RetrievalError("Duplicate chunk ID has conflicting content or provenance")
                if old is None or old.retrieval_score < hit.retrieval_score:
                    by_id[cid] = hit.model_copy(deep=True)
                fusion[cid] = fusion.get(cid, 0) + 1 / (60 + hit.rank)
        if multi:
            ordered = sorted(by_id.values(), key=lambda h: (-fusion[h.chunk.chunk_id], h.chunk.chunk_id))
        else:
            ordered = sorted(by_id.values(), key=lambda h: (-h.retrieval_score, h.chunk.chunk_id))
        # Limit merged candidates explicitly, even when many queries overlap little.
        ordered = ordered[:self.total_candidate_budget]
        if ranked and ordered:
            self.last_calls["rerank"] = 1
            try:
                scores = self.rerank(request.question, tuple(h.chunk.model_copy(deep=True) for h in ordered))
                if (not isinstance(scores, Sequence) or isinstance(scores, (str, bytes))
                        or len(scores) != len(ordered)):
                    raise ValueError("Expected one score per candidate")
                for hit, score in zip(ordered, scores):
                    hit.rerank_score = _number(score)
            except Exception:
                raise RetrievalError("Rerank failed") from None
            ordered.sort(key=lambda h: (-h.rerank_score, h.chunk.chunk_id))
        return [h.model_copy(update={"rank": rank})
                for rank, h in enumerate(ordered[:k], 1)]
