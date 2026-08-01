from langchain_core.prompts import ChatPromptTemplate
from app.generation.llm import llm

multi_query_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Generate 3 different rephrasings of the following question to help retrieve relevant documents from a vector database. Each version should approach the question from a slightly different angle. Provide only the 3 questions, one per line, no numbering, no extra text.

Original question: {question}
""")

def generate_queries(question):
    chain = multi_query_prompt | llm
    response = chain.invoke({"question": question})
    queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    return [question] + queries  # include original + 3 rephrasings