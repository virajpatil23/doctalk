from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_store import build_bm25_index

docs = load_pdf("data/sample_pdfs/sample.pdf")
chunks = chunk_documents(docs)
build_bm25_index(chunks)
print(f"BM25 index built with {len(chunks)} chunks")