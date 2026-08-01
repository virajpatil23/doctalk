from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. If the answer isn't in the context, say you don't know.

Context: {context}

Question: {question}
""")