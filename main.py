import streamlit as st
from document_handler import load_and_index_documents
from rag_pipeline import generate_answer
from evaluation import evaluate_all_metrics
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("📄 GenAI Document QA System")

uploaded_files = st.file_uploader("Upload files (.txt, .pdf, .docx)", accept_multiple_files=True, type=["txt", "pdf", "docx"])
query = st.text_input("Ask a question based on the uploaded documents:")

@st.cache_resource
def cached_indexing(files):
    return load_and_index_documents(files)


if st.button("Run RAG"):
    if not uploaded_files or not query:
        st.warning("Please upload files and enter a question.")
    else:
        with st.spinner("Indexing and generating answer..."):
            try:
                retriever, docs = cached_indexing(uploaded_files)
                answer, context_docs = generate_answer(query, retriever)

                st.subheader("🧠 Answer")
                st.write(answer)

                st.subheader("📚 Retrieved Context")
                for i, doc in enumerate(context_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.code(doc.page_content)

                st.subheader("📊 Evaluation Metrics")
                metrics = evaluate_all_metrics(query, answer, context_docs)
                st.json(metrics)
            except Exception as e:
                st.error(f"An error occurred: {e}")