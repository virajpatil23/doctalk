from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
import pickle
from rank_bm25 import BM25Okapi

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_semantic
from app.retrieval.vector_store import embedding_model, load_vector_store
from app.retrieval.bm25_store import build_bm25_index, load_bm25_index
from app.retrieval.hybrid import hybrid_search
from app.generation.llm import llm
from app.generation.prompts import rag_prompt
from langchain_community.vectorstores import FAISS

router = APIRouter()

UPLOAD_DIR = "data/sample_pdfs"
FAISS_PATH = "data/faiss_index_semantic"
BM25_PATH = "data/bm25_index_semantic.pkl"

# In-memory cache so we don't reload from disk on every request
_cache = {"vector_store": None, "bm25": None, "bm25_chunks": None}


def get_stores():
    if _cache["vector_store"] is None:
        _cache["vector_store"] = load_vector_store(path=FAISS_PATH)
    if _cache["bm25"] is None:
        _cache["bm25"], _cache["bm25_chunks"] = load_bm25_index(path=BM25_PATH)
    return _cache["vector_store"], _cache["bm25"], _cache["bm25_chunks"]


class ChatRequest(BaseModel):
    question: str


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docs = load_pdf(file_path)
    chunks = chunk_semantic(docs)

    vs = FAISS.from_documents(chunks, embedding_model)
    vs.save_local(FAISS_PATH)

    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    # Invalidate cache so the next /chat call picks up the new document
    _cache["vector_store"] = None
    _cache["bm25"] = None
    _cache["bm25_chunks"] = None

    return {"filename": file.filename, "chunks_created": len(chunks)}


@router.post("/chat")
async def chat(request: ChatRequest):
    vector_store, bm25, bm25_chunks = get_stores()

    if vector_store is None or bm25 is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Upload a PDF first.")

    retrieved = hybrid_search(request.question, vector_store, bm25, bm25_chunks, k=4)
    context = "\n\n".join([doc.page_content for doc in retrieved])

    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "question": request.question})

    sources = [doc.page_content[:200] + "..." for doc in retrieved]

    return {"answer": response.content, "sources": sources}