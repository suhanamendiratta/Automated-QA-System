from fastapi import FastAPI
from src.rag_pipeline import rag_query, initialize_pipeline

app = FastAPI()

@app.on_event("startup")
def startup_event():
    initialize_pipeline()


@app.get("/")
def home():
    return {"message": "QA System Running"}

@app.get("/ask")
def ask(q: str):
    return {"question": q, "answer": rag_query(q)}