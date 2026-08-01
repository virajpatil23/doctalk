from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.retrieval.vector_store import build_vector_store
from app.generation.llm import llm
from app.generation.prompts import rag_prompt

# 1. Load
docs = load_pdf("data/sample_pdfs/sample.pdf")
print(f"Loaded {len(docs)} pages")

# 2. Chunk
chunks = chunk_documents(docs)
print(f"Created {len(chunks)} chunks")

# 3. Embed + store
vector_store = build_vector_store(chunks)
print("Vector store built")

# 4. Query
question = "What is this document about?"
retrieved = vector_store.similarity_search(question, k=4)
context = "\n\n".join([doc.page_content for doc in retrieved])

chain = rag_prompt | llm
response = chain.invoke({"context": context, "question": question})
print("\nAnswer:", response.content)