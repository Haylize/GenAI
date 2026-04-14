
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectorstore import load_local_retriever
from qa_chain import get_medical_rag_chain
from memory.memory import ConversationMemory


def main():
    """Main execution loop for the interactive terminal medical assistant."""
    try:
        print("--- Initializing Medical RAG System ---")

        retriever = load_local_retriever()
        rag_chain = get_medical_rag_chain(retriever)
        memory = ConversationMemory(max_exchanges=5)

        print("\n" + "-" * 50)
        print("MEDICAL ASSISTANT IS ONLINE")
        print("Ask your question (or type 'quit' to exit)")
        print("Type 'history' to display the conversation history")
        print("Type 'clear' to reset the conversation memory")
        print("-" * 50 + "\n")

        while True:
            user_input = input("Your Question: ").strip()

            if user_input.lower() in ["quit", "exit", "stop", "quitter"]:
                print("Closing the assistant. Stay safe!")
                break

            if not user_input:
                continue

            if user_input.lower() == "history":
                memory.display()
                continue

            if user_input.lower() == "clear":
                memory.clear()
                continue

            print("\nAnalyzing knowledge base and generating response...")

            # Build the input dict with current question and conversation history
            chain_input = {
                "question": user_input,
                "history": memory.get_langchain_messages(),
            }

            # Run the RAG chain
            answer = rag_chain.invoke(chain_input)

            # Retrieve source documents for display
            docs = retriever.invoke(user_input)

            print("\n--- Assistant's Response ---")
            print(answer)

            print("\n--- Referenced Sources ---")
            if docs:
                seen = set()
                for doc in docs:
                    title = doc.metadata.get("disease_title", "Unknown Disease")
                    if title not in seen:
                        print(f"- {title}")
                        seen.add(title)
            else:
                print("- No specific documents found.")

            print("\n" + "-" * 50 + "\n")

            # Save the exchange to memory
            memory.add_exchange(user_input, answer)

    except Exception as e:
        print(f"System Error: {e}")
        print("Please verify that Ollama is running and the FAISS index exists.")


if __name__ == "__main__":
    main()
