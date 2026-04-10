from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    """Combines the page content of retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_medical_rag_chain(retriever):
    """
    Configures the local LLM and builds the RAG chain with conversation memory.

    The chain expects a dict with:
        - "question": the current user message (str)
        - "history": the conversation history (list of LangChain messages)

    Args:
        retriever: A LangChain retriever object (from vectorstore.py).

    Returns:
        A runnable RAG chain.
    """
    llm = ChatOllama(model="mistral", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a professional Medical Assistant.
Use the following pieces of retrieved context to answer the user's question.
If the answer is not contained within the context, clearly state that you do not know.
Always advise the user to consult a doctor for a formal diagnosis.

STRICT RULE: Always respond in the same language as the user's question.

Context:
{context}"""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "history": lambda x: x["history"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
