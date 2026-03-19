from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub

from src.embeddings import get_embedding
from src.utils import format_docs

loader = TextLoader("data/data.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embedding = get_embedding()

db = Chroma.from_documents(docs, embedding, persist_directory="chroma_db")
retriever = db.as_retriever()

# ⚠️ Add your HuggingFace API token in environment variable
llm = HuggingFaceHub(repo_id="google/flan-t5-base")

def rag_query(query):
    retrieved_docs = retriever.get_relevant_documents(query)
    context = format_docs(retrieved_docs)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    return llm(prompt)