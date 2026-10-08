"""골든셋 질문으로 질의 확장 결과를 눈으로 확인 (Dense 불필요, LLM만 사용)

전체 질문을 한 번의 요청으로 확장하고 평가와 같은 캐시를 쓴다 (첫 실행 1회, 재실행 0회).
실행: uv run python -m evaluation.check_expansion
"""

import json
import time

from evaluation.run_eval import (DEFAULT_GOLDEN, RESULT_DIR, _cache, _expansion_key, limiter, load_golden,
                                 prepare, save_caches, setup_cache)


def main():
    golden = load_golden(DEFAULT_GOLDEN)
    setup_cache(enabled=True)
    start = time.perf_counter()
    info = prepare(golden, ["multi_query"])  # 질의 확장 묶음 1회 + 임베딩 묶음 1회
    save_caches()
    print(f"MonoRouter 요청 {limiter.count}회, {round(time.perf_counter() - start, 2)}초, {info}")

    rows = []
    cache = _cache("expansions")
    for item in golden:
        queries = cache.get(_expansion_key(item["question"])) or [item["question"]]
        rows.append({"id": item["id"], "type": item["type"], "answer_articles": item["answer_articles"],
                     "queries": queries})
        print(f"[{item['id']}] ({item['type']}) 정답 {item['answer_articles']}")
        for i, q in enumerate(queries):
            print(f"   {'원' if i == 0 else i}: {q}")

    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / f"expansion_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
