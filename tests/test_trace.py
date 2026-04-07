def test_trace():
    print("starting", flush=True)
    try:
        print("loading TextLoader", flush=True)
        from langchain_community.document_loaders import TextLoader
        print("loading TextSplitter", flush=True)
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("loading FAISS", flush=True)
        from langchain_community.vectorstores import FAISS
        print("loading HuggingFacePipeline", flush=True)
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        print("loading embeddings", flush=True)
        from src.embeddings import get_embedding
        print("loading utils", flush=True)
        from src.utils import format_docs
        print("done", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
