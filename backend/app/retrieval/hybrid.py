def reciprocal_rank_fusion(result_lists, k=60):
    """
    result_lists: list of ranked document lists (e.g. [faiss_results, bm25_results])
    k: RRF constant (60 is the standard default from the original paper)
    Returns: documents ranked by fused score, deduplicated by content
    """
    scores = {}
    doc_lookup = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            key = doc.page_content  # use content as dedup key
            doc_lookup[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_lookup[key] for key in ranked_keys]


def hybrid_search(query, vector_store, bm25, bm25_chunks, k=4):
    faiss_results = vector_store.similarity_search(query, k=k)
    bm25_results = bm25_search_helper(query, bm25, bm25_chunks, k=k)
    fused = reciprocal_rank_fusion([faiss_results, bm25_results])
    return fused[:k]


def bm25_search_helper(query, bm25, chunks, k=4):
    from app.retrieval.bm25_store import bm25_search
    return bm25_search(query, bm25, chunks, k=k)

def multi_query_search(question, vector_store, k=4):
    from app.retrieval.multi_query import generate_queries
    queries = generate_queries(question)
    result_lists = [vector_store.similarity_search(q, k=k) for q in queries]
    fused = reciprocal_rank_fusion(result_lists)
    return fused[:k]