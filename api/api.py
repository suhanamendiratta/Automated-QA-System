from fastapi import FastAPI
from src.rag_pipeline import rag_query

app = FastAPI()

@app.get("/")
def home():
    return {"message": "QA System Running"}

@app.get("/ask")
def ask(q: str):
    return {"question": q, "answer": rag_query(q)}