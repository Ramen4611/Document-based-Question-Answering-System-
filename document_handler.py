from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import tempfile
import os
from PyPDF2 import PdfReader
import docx


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(file) -> str:
    return file.read().decode("utf-8")


def load_and_index_documents(uploaded_files):
    raw_docs = []

    for file in uploaded_files:
        filename = file.name.lower()
        ext = os.path.splitext(filename)[-1]

        if ext == ".pdf":
            text = extract_text_from_pdf(file)
        elif ext == ".docx":
            text = extract_text_from_docx(file)
        elif ext == ".txt":
            text = extract_text_from_txt(file)
        else:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding="utf-8") as tmp_file:
            tmp_file.write(text)
            tmp_file_path = tmp_file.name

        loader = TextLoader(tmp_file_path, encoding="utf-8")
        raw_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return retriever, docs
