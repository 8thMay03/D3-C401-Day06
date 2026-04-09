from langchain_core.vectorstores import VectorStoreRetriever

from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store
from config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVER_TOP_K


def get_retriever(top_k: int = RETRIEVER_TOP_K) -> VectorStoreRetriever:
    """Build a retriever backed by the persisted Chroma vector store."""
    embedding_model = get_embedding_model(EMBEDDING_MODEL)
    vector_store = load_vector_store(embedding_model, CHROMA_DIR)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
