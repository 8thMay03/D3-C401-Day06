from langchain_openai import OpenAIEmbeddings


def get_embedding_model(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """Return an OpenAI embedding model instance."""
    return OpenAIEmbeddings(model=model)
