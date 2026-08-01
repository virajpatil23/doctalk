from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import EMBEDDING_MODEL, FAISS_INDEX_PATH
import os

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

def build_vector_store(chunks):
    vs = FAISS.from_documents(chunks, embedding_model)
    vs.save_local(FAISS_INDEX_PATH)
    return vs

def load_vector_store(path=None):
    target_path = path or FAISS_INDEX_PATH
    if os.path.exists(target_path):
        return FAISS.load_local(target_path, embedding_model, allow_dangerous_deserialization=True)
    return None