import sys
import os

# Ensure the script can import local modules from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectorstore import load_local_retriever
from qa_chain import get_medical_rag_chain

def main():
    """Main execution loop for the interactive terminal medical assistant."""
    try:
        print("--- Initializing Medical RAG System ---")
        
        # Initialize modular components
        retriever = load_local_retriever()
        rag_chain = get_medical_rag_chain(retriever)
        
        print("\n" + "="*50)
        print("MEDICAL ASSISTANT IS ONLINE")
        print("Ask your question (or type 'quit' to exit)")
        print("="*50 + "\n")

        while True:
            user_input = input("Your Question: ")
            
            # Handle exit commands
            if user_input.lower() in ["quit", "exit", "stop", "quitter"]:
                print("Closing the assistant. Stay safe!")
                break
                
            if not user_input.strip():
                continue

            print("\n🔍 Analyzing knowledge base and generating response...")
            
            # Execute the RAG chain and retrieve documents for sources
            answer = rag_chain.invoke(user_input)
            docs = retriever.invoke(user_input)
            
            print("\n--- Assistant's Response ---")
            print(answer)
            
            print("\n--- Referenced Sources ---")
            if docs:
                for doc in docs:
                    title = doc.metadata.get('disease_title', 'Unknown Disease')
                    print(f"- {title}")
            else:
                print("- No specific documents found.")
            
            print("\n" + "-"*50 + "\n")

    except Exception as e:
        print(f"System Error: {e}")
        print("Please verify that Ollama is running and the FAISS index exists.")

if __name__ == "__main__":
    main()