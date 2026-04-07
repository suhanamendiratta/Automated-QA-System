from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding
import pytest

def test_faiss():
    loader = TextLoader("data/data.txt")
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())
    embedding = get_embedding()
    
    # Just index first two documents to be fast in tests
    db = FAISS.from_documents(docs[:2], embedding)
    
    retriever = db.as_retriever()
    results = retriever.invoke("What is this?")
    assert len(results) > 0
    assert results[0].page_content is not None
