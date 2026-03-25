import traceback
try:
    from src.rag_pipeline import retriever, llm, format_docs
    
    query = "What is the capital of France?"
    print(f"QUERY: {query}")
    
    docs = retriever.invoke(query)
    context = format_docs(docs)
    print(f"\n--- RETRIEVED CONTEXT ---\n{context}\n-------------------------")
    
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    print(f"\n--- FULL PROMPT ---\n{prompt}\n-------------------")
    
    ans = llm.invoke(prompt)
    print(f"\n--- LLM ANSWER ---\n{ans}\n------------------")
    
except Exception as e:
    traceback.print_exc()
