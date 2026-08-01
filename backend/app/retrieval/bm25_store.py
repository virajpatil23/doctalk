from rank_bm25 import BM25Okapi
import pickle
import os

BM25_INDEX_PATH = "data/bm25_index.pkl"

def build_bm25_index(chunks):
    """chunks: list of LangChain Document objects"""
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    return bm25, chunks

def load_bm25_index(path=None):
    target_path = path or BM25_INDEX_PATH
    if not os.path.exists(target_path):
        return None, None
    with open(target_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]

def bm25_search(query, bm25, chunks, k=4):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked_indices[:k]]