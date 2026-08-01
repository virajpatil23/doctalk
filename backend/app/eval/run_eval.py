print("FILE LOADING")

import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from app.retrieval.vector_store import load_vector_store
from app.retrieval.bm25_store import load_bm25_index
from app.retrieval.hybrid import hybrid_search, multi_query_search
from app.eval.metrics import is_hit, reciprocal_rank, compute_metrics

def run_eval(strategy_name="baseline", k=4):
    if strategy_name == "hybrid_semantic_chunks":
        vector_store = load_vector_store(path="data/faiss_index_semantic")
        bm25, bm25_chunks = load_bm25_index(path="data/bm25_index_semantic.pkl")
    else:
        vector_store = load_vector_store()
        bm25, bm25_chunks = load_bm25_index()

    if vector_store is None:
        raise ValueError(f"No vector store found for strategy: {strategy_name}")

    eval_path = os.path.join(os.path.dirname(__file__), "eval_set.json")
    with open(eval_path, "r") as f:
        eval_set = json.load(f)

    results = []
    for item in eval_set:
        if strategy_name in ("hybrid", "hybrid_semantic_chunks"):
            if bm25 is None:
                raise ValueError("No BM25 index found for this strategy.")
            retrieved = hybrid_search(item["question"], vector_store, bm25, bm25_chunks, k=k)
        elif strategy_name == "multi_query":
            retrieved = multi_query_search(item["question"], vector_store, k=k)
        else:
            retrieved = vector_store.similarity_search(item["question"], k=k)

        hit = is_hit(retrieved, item["expected_keywords"])
        rr = reciprocal_rank(retrieved, item["expected_keywords"])
        results.append({"question": item["question"], "hit": hit, "rr": rr})

    metrics = compute_metrics(results)
    print(f"\n=== {strategy_name} ===")
    print(f"Hit Rate: {metrics['hit_rate']:.2%}")
    print(f"MRR: {metrics['mrr']:.3f}")

    for r in results:
        status = "✓" if r["hit"] else "✗"
        print(f"  {status} {r['question']}")

    return metrics

if __name__ == "__main__":
    print("Script started")
    baseline_metrics = run_eval(strategy_name="baseline")
    hybrid_metrics = run_eval(strategy_name="hybrid")
    multi_query_metrics = run_eval(strategy_name="multi_query")
    hybrid_semantic_metrics = run_eval(strategy_name="hybrid_semantic_chunks")

    print("\n=== COMPARISON ===")
    print(f"Baseline              : Hit Rate {baseline_metrics['hit_rate']:.2%} | MRR {baseline_metrics['mrr']:.3f}")
    print(f"Hybrid                : Hit Rate {hybrid_metrics['hit_rate']:.2%} | MRR {hybrid_metrics['mrr']:.3f}")
    print(f"Multi-Query           : Hit Rate {multi_query_metrics['hit_rate']:.2%} | MRR {multi_query_metrics['mrr']:.3f}")
    print(f"Hybrid+Semantic Chunk : Hit Rate {hybrid_semantic_metrics['hit_rate']:.2%} | MRR {hybrid_semantic_metrics['mrr']:.3f}")