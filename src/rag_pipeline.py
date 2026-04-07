from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline

from src.embeddings import get_embedding
from src.utils import format_docs

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

retriever = None
llm = None

def initialize_pipeline():
    global retriever, llm
    if retriever is not None and llm is not None:
        return

    print("Starting data load", flush=True)
    loader = TextLoader("data/data.txt")
    documents = loader.load()

    print("Splitting text", flush=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    print("Getting embedding", flush=True)
    embedding = get_embedding()

    print("Creating FAISS DB", flush=True)
    db = FAISS.from_documents(docs, embedding)
    retriever = db.as_retriever()

    print("Initializing LLM locally (no API token required)", flush=True)
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150)
    llm = HuggingFacePipeline(pipeline=pipe)
    print("Pipeline initialized", flush=True)

def rag_query(query):
    if retriever is None or llm is None:
        initialize_pipeline()
        
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)
    prompt = f"Use the following piece of context to answer the question. If the answer is not in the context, say 'I cannot answer this based on the context'.\nContext: {context}\nQuestion: {query}\nAnswer:"
    return llm.invoke(prompt)