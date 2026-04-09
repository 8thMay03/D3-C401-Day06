from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT
from src.retriever import get_retriever


def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    """Assemble the full RAG chain: retriever → prompt → LLM → output."""
    retriever = get_retriever()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
