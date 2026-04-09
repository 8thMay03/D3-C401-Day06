from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def create_vector_store(
    chunks: list,
    embedding_model: Embeddings,
    persist_directory: Path,
) -> Chroma:
    """Create (or overwrite) a Chroma vector store from document chunks."""
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(persist_directory),
        collection_name="xanhsm_policies",
    )
    print(f"Vector store created at {persist_directory} with {len(chunks)} chunks")
    return vector_store


def load_vector_store(
    embedding_model: Embeddings,
    persist_directory: Path,
) -> Chroma:
    """Load an existing Chroma vector store from disk."""
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embedding_model,
        collection_name="xanhsm_policies",
    )
