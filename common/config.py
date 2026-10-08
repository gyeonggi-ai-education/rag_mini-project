import os

from dotenv import load_dotenv

load_dotenv(override=True)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("MONOROUTER_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("MONOROUTER_BASE_URL")

MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
TEMPERATURE = os.getenv("LLM_TEMPERATURE", 2)
MAX_TOKENS = os.getenv("LLM_MAX_TOKENS", 2086)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

# MonoRouter가 분당 30회 제한이고 키를 함께 쓰므로 클라이언트 재시도는 1회로 줄인다(C와 합의). 429 대기·재시도는 평가 실행기가 맡는다.
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 1))

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)


# 검색 설정. API와 평가 실행기가 같은 값을 쓰도록 여기서만 정의한다.
# 컬렉션은 QDRANT_COLLECTION 환경변수로 바꿀 수 있다. 기본값은 팀 공유 스냅샷(46청크) 컬렉션
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "data_ingestion_law21311_v1")
CANDIDATE_K = 20
FINAL_K = 5
