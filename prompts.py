from langchain.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't know.".

Context:
{context}

Question:
{question}

Answer:
"""
)
