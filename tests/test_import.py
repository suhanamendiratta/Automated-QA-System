def test_import():
    import traceback
    try:
        from src.rag_pipeline import rag_query
        print("Import OK")
        print("Testing query...")
        print(rag_query("What is this dataset about?"))
        print("Success")
    except Exception as e:
        traceback.print_exc()
        raise e
