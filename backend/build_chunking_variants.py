from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_fixed, chunk_semantic
from app.retrieval.vector_store import embedding_model
from app.retrieval.bm25_store import build_bm25_index
from langchain_community.vectorstores import FAISS

docs = load_pdf("data/sample_pdfs/sample.pdf")

# Fixed-size (existing baseline) — already built, skip re-saving over it
fixed_chunks = chunk_fixed(docs)
print(f"Fixed chunks: {len(fixed_chunks)}")

# Semantic/paragraph-oriented
semantic_chunks = chunk_semantic(docs)
print(f"Semantic chunks: {len(semantic_chunks)}")

# Build separate FAISS index for semantic chunks
semantic_vs = FAISS.from_documents(semantic_chunks, embedding_model)
semantic_vs.save_local("data/faiss_index_semantic")
print("Semantic FAISS index built")

# Build separate BM25 index for semantic chunks
tokenized_corpus = [doc.page_content.lower().split() for doc in semantic_chunks]
import pickle
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tokenized_corpus)
with open("data/bm25_index_semantic.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "chunks": semantic_chunks}, f)
print("Semantic BM25 index built")