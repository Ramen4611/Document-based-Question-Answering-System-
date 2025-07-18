from langchain.evaluation.qa import QAEvalChain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def evaluate_all_metrics(query, answer, context_docs):
    context = "\n".join([doc.page_content for doc in context_docs])
    inputs = {
        "question": query,
        "context": context,
        "answer": answer,
        "reference": "(based on context)"
    }

    metrics_prompt = f"""
    Given the following QA pair:

    Question: {query}
    Answer: {answer}
    Context: {context}

    Score the following from 1 (low) to 5 (high):
    1. Faithfulness
    2. Answer Relevance
    3. Context Precision
    4. Context Recall

    Return a JSON like:
    {{
      "faithfulness": ..., 
      "answer_relevance": ..., 
      "context_precision": ..., 
      "context_recall": ...
    }}
    """

    eval_response = llm.predict(metrics_prompt)
    try:
        import json
        return json.loads(eval_response)
    except:
        return {"error": "Failed to parse LLM response", "raw": eval_response}
