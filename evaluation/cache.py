"""MonoRouter 호출을 줄이기 위한 평가용 캐시 (메모리 + 디스크)

같은 입력이면 다시 호출하지 않는다.
- 임베딩: 같은 질문을 여러 모드(baseline/rerank/multi_query/combined)가 각각 임베딩하던 중복 제거,
          Multi Query 질의 여러 개는 한 번의 요청으로 묶어서 임베딩 (prefetch)
- 질의 확장: multi_query와 combined가 같은 확장 질의를 쓰도록 (비교 공정성 + 호출 절감)
- 답변 생성·채점: 같은 질문 + 같은 청크 조합이면 재사용

키에 모델명·프롬프트 버전을 포함하므로, 프롬프트나 모델을 바꾸면 자동으로 새로 호출한다.
캐시 파일: evaluation/.cache/*.json (git 제외). 새로 호출하려면 --no-cache 또는 폴더 삭제.
"""

import hashlib
import json
import threading
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


class JsonCache:
    def __init__(self, name: str, enabled: bool = True):
        self.path = CACHE_DIR / f"{name}.json"
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._data: dict = {}
        if enabled and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    @staticmethod
    def key(*parts) -> str:
        raw = json.dumps(parts, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str):
        if not self.enabled:
            return None
        value = self._data.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, key: str, value):
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = value

    def save(self):
        if not self.enabled:
            return
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


class CachedEmbeddings:
    """retrieval/dense.py 의 임베딩 객체를 감싸 같은 문장은 한 번만 임베딩한다.

    B의 파일은 바꾸지 않고, 평가 실행 중에만 dense 모듈의 _embedding 을 이 객체로 바꿔 끼운다.
    """

    def __init__(self, inner, cache: JsonCache, model_name: str, call=None):
        """call: 실제 요청을 감싸는 함수 (호출 제한 + 429 재시도, evaluation.rate_limit.call_limited)"""
        self.inner = inner
        self.cache = cache
        self.model_name = model_name
        self.call = call or (lambda fn, *a: fn(*a))
        self.requests = 0  # 실제로 MonoRouter에 보낸 임베딩 요청 수

    def _key(self, text: str) -> str:
        return JsonCache.key("embedding", self.model_name, text)

    def prefetch(self, texts: list[str]):
        """캐시에 없는 문장들을 한 번의 요청으로 임베딩 (Multi Query 질의 묶음용)"""
        missing = list(dict.fromkeys(t for t in texts if self.cache.get(self._key(t)) is None))
        if not missing:
            return
        self.requests += 1
        for text, vector in zip(missing, self.call(self.inner.embed_documents, missing)):
            self.cache.set(self._key(text), vector)

    def embed_query(self, text: str) -> list[float]:
        vector = self.cache.get(self._key(text))
        if vector is None:
            self.requests += 1
            vector = self.call(self.inner.embed_query, text)
            self.cache.set(self._key(text), vector)
        return vector

    def __getattr__(self, name):
        return getattr(self.inner, name)
