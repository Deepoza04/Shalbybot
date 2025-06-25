import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

PDF_DIR = "pdfs"
CHROMA_DIR = "chroma_store"

def get_retrieval_answer(query):
    if not os.path.exists(CHROMA_DIR):
        return None  # No DB yet

    try:
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embed)
        retriever = vectordb.as_retriever()
        docs = retriever.get_relevant_documents(query)
        if docs:
            return docs[0].page_content
        else:
            return "I searched the PDFs but found no relevant answer."
    except:
        return None