def is_hit(retrieved_docs, expected_keywords):
    """Check if any retrieved chunk contains at least one expected keyword."""
    combined_text = " ".join([doc.page_content.lower() for doc in retrieved_docs])
    return any(keyword.lower() in combined_text for keyword in expected_keywords)

def reciprocal_rank(retrieved_docs, expected_keywords):
    """Find the rank (1-indexed) of the first chunk containing an expected keyword. Returns 0 if none found."""
    for i, doc in enumerate(retrieved_docs):
        text = doc.page_content.lower()
        if any(keyword.lower() in text for keyword in expected_keywords):
            return 1 / (i + 1)
    return 0

def compute_metrics(results):
    """results: list of dicts with 'hit' (bool) and 'rr' (float) per question."""
    hit_rate = sum(r["hit"] for r in results) / len(results)
    mrr = sum(r["rr"] for r in results) / len(results)
    return {"hit_rate": hit_rate, "mrr": mrr}