# DocTalk — Retrieval Strategy Evaluation

Evaluated against a 22-question test set derived from a sample clinical NLP / ICD-10 mapping PDF (45 pages).

**Metrics:**
- **Hit Rate** — % of questions where a correct chunk appeared anywhere in the top-4 retrieved results
- **MRR (Mean Reciprocal Rank)** — rewards correct chunks appearing higher in the ranking, not just present

## Results

| Strategy | Chunking | Hit Rate | MRR | Notes |
|---|---|---|---|---|
| Baseline (FAISS dense retrieval) | Fixed (500/50, 174 chunks) | 81.82% | 0.674 | Pure semantic similarity search |
| Hybrid (BM25 + FAISS, RRF fusion) | Fixed (500/50, 174 chunks) | 81.82% | 0.720 | Same hit rate, better ranking |
| Multi-Query (RAG-Fusion) | Fixed (500/50, 174 chunks) | 81.82% | ~0.65 | Underperformed baseline; some run-to-run variance since it depends on live LLM rephrasing |
| **Hybrid + Semantic Chunking** | Larger, paragraph-oriented (800/100, 132 chunks) | **90.91%** | **0.856** | **Best overall** — biggest single improvement of any technique tested |

## Key findings

**Chunking strategy mattered more than retrieval algorithm choice.** Switching from fixed 500-character chunks to larger (~800 char), paragraph-boundary-respecting chunks — combined with hybrid search — produced the largest jump in both metrics of anything tested (+9 points hit rate, +0.18 MRR over baseline). This suggests the original fixed-size chunking was splitting relevant explanations across chunk boundaries, losing context that larger, more naturally-bounded chunks preserved.

**Hybrid search improved ranking quality without changing hit rate (on fixed chunking).** Combining BM25 keyword search with FAISS dense retrieval via Reciprocal Rank Fusion didn't surface new correct chunks on the original chunking, but pushed correct chunks higher in the ranking on average (MRR +6.8% over baseline).

**Multi-query / RAG-Fusion underperformed on this document.** Generating 3 LLM-rephrased versions of each question and fusing results reduced MRR versus the plain baseline. Likely explanation: this technique is designed for ambiguous queries or broad/diverse corpora where different phrasings surface genuinely different relevant content. On a narrow, single-topic technical document, the rephrasings introduced retrieval noise rather than new signal. This technique also showed run-to-run score variance, since it depends on live, non-deterministic LLM calls — unlike the other strategies, which are fully deterministic given the same chunks.

**Takeaway:** for a focused technical document like this one, chunking strategy is the highest-leverage lever to pull, ahead of adding retrieval algorithm complexity. This is a useful, generalizable finding — not every advanced RAG technique adds value on every corpus, and testing against a baseline (rather than assuming "more sophisticated = better") is what surfaces that.

## Setup

- **Embeddings:** BAAI/bge-small-en-v1.5 (local, sentence-transformers)
- **LLM:** Groq — openai/gpt-oss-120b
- **Vector store:** FAISS
- **Keyword search:** BM25 (rank_bm25)
- **Fusion method:** Reciprocal Rank Fusion (k=60)
- **Retrieval depth:** top-4 chunks per query