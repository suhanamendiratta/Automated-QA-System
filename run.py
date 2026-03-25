import sys
print("Interpreter started!", flush=True)
print("Importing rag_pipeline...", flush=True)
from src.rag_pipeline import rag_query
print("Imported rag_pipeline successfully!", flush=True)
if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        print(rag_query(q))