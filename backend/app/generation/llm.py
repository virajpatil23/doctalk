from langchain_groq import ChatGroq
from app.config import LLM_MODEL

llm = ChatGroq(model=LLM_MODEL, temperature=0)