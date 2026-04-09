from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents(data_dir: Path) -> list:
    """Load all .txt policy documents from the data directory."""
    loader = DirectoryLoader(
        str(data_dir),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {data_dir}")
    return documents
