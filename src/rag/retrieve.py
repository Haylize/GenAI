import os
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION & PATHS ---
# Calculate the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
# Must match the model used during the ingestion phase
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def format_docs(docs):
    """Combines the content of retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def load_medical_system():
    """Initializes the FAISS index and the local LLM (Mistral)."""
    print("--- Loading FAISS Index ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # allow_dangerous_deserialization is required for loading local pickle-based FAISS files
    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print("--- Connecting to Local LLM (Mistral via Ollama) ---")
    # Temperature 0 ensures factual, non-creative responses
    llm = ChatOllama(model="mistral", temperature=0)
    
    return vectorstore, llm

def run_rag_query(query: str, vectorstore, llm):
    """
    Orchestrates the RAG process using LCEL (LangChain Expression Language).
    Retrieves context, formats the prompt, and generates an answer.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Strict System Prompt to ensure safety and relevance
    template = """You are a professional Medical Assistant. 
Use the following pieces of retrieved context to answer the user's question. 
If the answer is not contained within the context, clearly state that you do not know. 
Always advise the user to consult a doctor for a formal diagnosis.

Context:
{context}

Question: {question}
Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Building the LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return both the AI response and the source documents for transparency
    return rag_chain.invoke(query), retriever.invoke(query)

if __name__ == "__main__":
    # 1. System initialization
    try:
        db, ai_model = load_medical_system()
        
        print("\n" + "="*50)
        print("🤖 MEDICAL ASSISTANT IS ONLINE")
        print("Type your question below (or type 'quit' to exit)")
        print("="*50 + "\n")

        # 2. Interactive Chat Loop
        while True:
            user_input = input("👉 Your Question: ")
            
            # Exit conditions
            if user_input.lower() in ["quit", "exit", "stop", "quitter"]:
                print("Closing the assistant. Stay safe! 👋")
                break
                
            if not user_input.strip():
                continue

            print("\n🔍 Searching knowledge base and generating answer...")
            
            # 3. Process the query
            answer, docs = run_rag_query(user_input, db, ai_model)
            
            print("\n--- 🩺 Assistant's Response ---")
            print(answer)
            
            # 4. Display metadata for verification
            print("\n--- 📚 Referenced Sources ---")
            if docs:
                for doc in docs:
                    title = doc.metadata.get('disease_title', 'Unknown Title')
                    source = doc.metadata.get('source_file', 'Unknown File')
                    print(f"- {title} (Source: {source})")
            else:
                print("- No specific sources found in the database.")
            
            print("\n" + "-"*50 + "\n")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Check if Ollama is running and your 'faiss_index' folder exists.")