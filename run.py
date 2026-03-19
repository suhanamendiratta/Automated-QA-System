from src.rag_pipeline import rag_query

if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        print(rag_query(q))