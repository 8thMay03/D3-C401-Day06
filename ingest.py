"""
Ingest pipeline: load policy documents → chunk → embed → store in ChromaDB.
Run this script once (or whenever documents change) before starting the app.

Usage:
    python ingest.py
"""

from dotenv import load_dotenv

load_dotenv()

from config import DATA_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store


def main():
    print("=== XanhSM Policy Ingestion Pipeline ===\n")

    print("[1/4] Loading documents...")
    documents = load_documents(DATA_DIR)

    print("\n[2/4] Splitting into chunks...")
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    print("\n[3/4] Initializing embedding model...")
    embedding_model = get_embedding_model(EMBEDDING_MODEL)

    print("\n[4/4] Creating vector store...")
    create_vector_store(chunks, embedding_model, CHROMA_DIR)

    print("\n=== Ingestion complete! ===")


if __name__ == "__main__":
    main()
