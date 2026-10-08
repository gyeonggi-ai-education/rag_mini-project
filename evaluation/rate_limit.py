"""MonoRouter 호출 속도 제한 (분당 30회 제한 대응)

임베딩(Dense 검색)과 LLM 호출이 모두 MonoRouter를 거치므로, 평가에서 호출하기 직전에 acquire()로
최근 60초 호출 수를 확인하고 한도에 닿으면 기다린다. 팀원이 같은 키를 동시에 쓰면 한도를 나눠 쓰게 되므로
기본값은 여유를 두고 분당 25회로 둔다 (환경변수 EVAL_RPM 으로 변경).
"""

import os
import threading
import time
from collections import deque

RPM = int(os.getenv("EVAL_RPM", "25"))
WINDOW_SEC = 60.0


class RateLimiter:
    def __init__(self, rpm: int = RPM):
        self.rpm = rpm
        self.calls: deque[float] = deque()
        self.waited_sec = 0.0  # 제한 때문에 기다린 누적 시간 (지연 해석용)
        self.count = 0  # 실제로 보낸 요청 수 (캐시 적중은 포함 안 됨)
        self.rate_limited = 0  # 429를 받은 횟수
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] >= WINDOW_SEC:
                self.calls.popleft()
            if len(self.calls) >= self.rpm:
                wait = WINDOW_SEC - (now - self.calls[0]) + 0.1
                time.sleep(wait)
                self.waited_sec += wait
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= WINDOW_SEC:
                    self.calls.popleft()
            self.calls.append(now)
            self.count += 1


limiter = RateLimiter()

# 같은 키를 여러 사람이 함께 쓰면 내 호출을 제한해도 429가 날 수 있다 → 1분 기다렸다가 다시 시도
RATE_LIMIT_RETRIES = int(os.getenv("EVAL_RATE_LIMIT_RETRIES", "3"))
RATE_LIMIT_WAIT_SEC = 61.0


def is_rate_limit_error(error: BaseException) -> bool:
    return type(error).__name__ == "RateLimitError" or "429" in str(error)[:200]


def call_limited(fn, *args, **kwargs):
    """호출 제한(acquire) + 429면 1분 대기 후 재시도"""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        limiter.acquire()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not is_rate_limit_error(e) or attempt == RATE_LIMIT_RETRIES:
                raise
            limiter.rate_limited += 1
            time.sleep(RATE_LIMIT_WAIT_SEC)
            limiter.waited_sec += RATE_LIMIT_WAIT_SEC


def limited(fn):
    """함수 호출마다 call_limited 적용"""
    def wrapper(*args, **kwargs):
        return call_limited(fn, *args, **kwargs)
    wrapper.__wrapped__ = fn
    return wrapper
