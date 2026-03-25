import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from src.embeddings import get_embedding

print("loading data...")
loader = TextLoader("data/data.txt")
docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())

print("getting embedding...")
embedding = get_embedding()

print("calling from_documents...")
import traceback
try:
    db = Chroma.from_documents(docs[:2], embedding, persist_directory="chroma_db")
    print("success!")
except Exception as e:
    traceback.print_exc()
