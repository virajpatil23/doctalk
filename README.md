# DocTalk

A RAG (Retrieval-Augmented Generation) application that answers questions grounded in an uploaded PDF — built from scratch to benchmark and compare retrieval strategies rather than assuming one approach is best.

## What it does

Upload any PDF, then ask questions about it in plain English. The system retrieves the most relevant sections of the document and generates an answer using only that retrieved context — it explicitly declines to answer when the document doesn't contain the information, rather than hallucinating from general knowledge.

## Why this project

Most RAG tutorials stop at "it works." This project instead treats retrieval as an empirical question: I built an evaluation harness (22 question/answer pairs with hit-rate and MRR scoring) and used it to test four different retrieval strategies against each other, rather than assuming any one technique is automatically better.

## Results

| Strategy | Hit Rate | MRR |
|---|---|---|
| Baseline (FAISS dense retrieval) | 81.82% | 0.674 |
| Hybrid (BM25 + FAISS, RRF fusion) | 81.82% | 0.720 |
| Multi-Query (RAG-Fusion) | 81.82% | ~0.65 |
| **Hybrid + Semantic Chunking** | **90.91%** | **0.856** |

**Key finding:** chunking strategy had more impact than retrieval algorithm sophistication. Switching from fixed 500-character chunks to larger, paragraph-boundary-respecting chunks (combined with hybrid search) produced the biggest single improvement — bigger than adding BM25 fusion or multi-query rephrasing alone. Multi-query/RAG-Fusion actually underperformed the baseline on this narrow technical document, a useful reminder that more sophisticated techniques aren't universally better — they need to be tested against the actual corpus.

Full write-up: [`backend/results/eval_comparison.md`](backend/results/eval_comparison.md)

## Architecture
PDF Upload → Text Extraction → Chunking → Embedding → Vector Store (FAISS) + Keyword Index (BM25)
↓
User Question → Hybrid Retrieval (RRF fusion) → Top-K Chunks → LLM Generation → Answer + Sources

## Tech stack

- **Backend:** FastAPI, LangChain
- **Embeddings:** `sentence-transformers` (BAAI/bge-small-en-v1.5) — local, free
- **LLM:** Groq (`openai/gpt-oss-120b`) — free tier, fast inference
- **Vector search:** FAISS
- **Keyword search:** BM25 (`rank_bm25`)
- **Frontend:** React + Vite

## Running locally

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
# Add your Groq API key to backend/.env as GROQ_API_KEY=your_key
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`, upload a PDF, and start asking questions.

## Evaluation

To rerun the retrieval strategy comparison:
```bash
cd backend
python -m app.eval.run_eval
```

## Known limitations

- Retrieval quality is sensitive to how closely a question's phrasing matches the document's own language — vague questions on long, dense documents (e.g. legal text) can miss relevant sections that more specific phrasing retrieves correctly.
- No OCR support — scanned/image-based PDFs won't extract text.
- Multi-query/RAG-Fusion is included as a tested-and-rejected approach for this corpus type, kept in the codebase to show the comparison rather than silently dropped.