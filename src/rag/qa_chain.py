from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    """
    Combine retrieved documents into a readable context string,
    including source information for each chunk.
    """
    if not docs:
        return "No relevant context found."

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("disease_title") or doc.metadata.get("source_file", "Unknown source")
        content = doc.page_content.strip()

        formatted.append(
            f"[Source {i}: {source}]\n{content}"
        )

    return "\n\n".join(formatted)


def get_medical_rag_chain(retriever):
    """
    Configure the local LLM and build the RAG chain with conversation memory.

    The chain expects a dict with:
        - "question": the current user message (str)
        - "history": the conversation history (list of LangChain messages)

    Args:
        retriever: A LangChain retriever object (from vectorstore.py).

    Returns:
        A runnable RAG chain.
    """

    llm = ChatOllama(
        model="mistral",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a professional medical assistant.

Your role is to answer ONLY using the retrieved context provided below.

Rules:
- Use only the information explicitly present in the retrieved context.
- If the answer is not contained in the context, clearly say that you do not know based on the available documents.
- Do not invent symptoms, treatments, causes, or recommendations.
- Always respond in the same language as the user's question.
- Keep the answer clear, concise, and medically cautious.
- Always advise the user to consult a doctor or healthcare professional for a formal diagnosis.
- When relevant, mention the source(s) used.

STRICT LANGUAGE RULE:
- If the user writes in French, you MUST answer in French.
- If the user writes in English, you MUST answer in English.
- Never switch language.

Preferred answer structure:
1. Main answer
2. Additional useful details
3. Medical caution

Retrieved context:
{context}"""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "history": lambda x: x["history"],
            "question": lambda x: f"Réponds STRICTEMENT en français.\nQuestion: {x['question']}",
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain