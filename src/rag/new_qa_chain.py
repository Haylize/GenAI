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
    Optimized with Llama 3.2 3B for speed and precision.
    """

    # Utilisation de Llama 3.2 3B : beaucoup plus léger et rapide que Mistral 7B
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        num_predict=300, # Limite la longueur pour accélérer la génération
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a professional medical assistant.

Your role is to answer ONLY using the retrieved context provided below.

Rules:
- Use only the information explicitly present in the retrieved context.
- If the answer is not contained in the context, clearly say that you do not know.
- Do not invent symptoms, treatments, or recommendations.
- Always respond in the same language as the user's question.
- Keep the answer clear, concise, and medically cautious.
- Always advise the user to consult a healthcare professional.

STRICT LANGUAGE RULE:
- If the user writes in French, you MUST answer in French.
- If the user writes in English, you MUST answer in English.

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