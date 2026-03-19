from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

def get_embedding():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")