from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from prompts import QA_PROMPT

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

def generate_answer(query, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    result = qa_chain(query)
    answer = result['result']
    context_docs = result['source_documents']
    return answer, context_docs
