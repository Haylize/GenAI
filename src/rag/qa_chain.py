from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    """Combines the page content of retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_medical_rag_chain(retriever):
    """
    Configures the local LLM and builds the RAG chain using LCEL.
    """
    llm = ChatOllama(model="mistral", temperature=0)
    
    # We add a specific instruction to reply in the user's language
    template = """You are a professional Medical Assistant. 
Use the following pieces of retrieved context to answer the user's question. 
If the answer is not contained within the context, clearly state that you do not know. 
Always advise the user to consult a doctor for a formal diagnosis.

STRICT RULE: Always respond in the same language as the user's question.

Context:
{context}

Question: {question}
Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain