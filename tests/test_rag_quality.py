def test_rag_quality():
    import traceback
    try:
        import src.rag_pipeline as pipeline
        from src.utils import format_docs
        
        pipeline.initialize_pipeline()
        
        query = "What is the capital of France?"
        print(f"QUERY: {query}")
        
        docs = pipeline.retriever.invoke(query)
        context = format_docs(docs)
        print(f"\n--- RETRIEVED CONTEXT ---\n{context}\n-------------------------")
        
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        print(f"\n--- FULL PROMPT ---\n{prompt}\n-------------------")
        
        ans = pipeline.llm.invoke(prompt)
        print(f"\n--- LLM ANSWER ---\n{ans}\n------------------")
        
    except Exception as e:
        traceback.print_exc()
        raise e
