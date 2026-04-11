import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from langchain_ollama import ChatOllama

from rag.vectorstore import load_local_retriever
from rag.qa_chain import get_medical_rag_chain
from memory.memory import ConversationMemory
from agents.tools.calculator import calculator_tool
from agents.tools.weather import weather_tool
from agents.tools.web_search import web_search_tool


class AssistantRouter:
    """
    Simple rule-based router for the hybrid assistant.

    Routes user questions to:
    - RAG for medical/document questions
    - calculator tool
    - weather tool
    - web search tool
    - normal LLM chat fallback
    """

    def __init__(self):
        self.retriever = load_local_retriever()
        self.rag_chain = get_medical_rag_chain(self.retriever)
        self.memory = ConversationMemory(max_exchanges=5)
        self.chat_llm = ChatOllama(model="mistral", temperature=0)

        self.medical_keywords = [
            "grippe", "migraine", "angine", "rhume", "otite", "bronchite",
            "sinusite", "gastro", "conjonctivite", "allergie", "infection urinaire",
            "stress aigu", "intoxication alimentaire", "symptome", "symptômes",
            "traitement", "maladie", "fièvre", "toux", "courbatures", "maux de tête"
        ]

        self.weather_keywords = [
            "météo", "meteo", "température", "temperature", "pluie",
            "soleil", "temps à", "temps de", "il fait quel temps"
        ]

        self.web_keywords = [
            "recherche web", "cherche sur le web", "cherche en ligne",
            "internet", "web", "actualité", "actualités", "news"
        ]

    def is_calculation(self, question: str) -> bool:
        """
        Detect simple math expressions or calculator-like questions.
        """
        q = question.lower().strip()

        if any(word in q for word in ["calcule", "calcul", "combien fait"]):
            return True

        return bool(re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.]+", q))

    def is_weather_question(self, question: str) -> bool:
        q = question.lower()
        return any(keyword in q for keyword in self.weather_keywords)

    def is_web_search_question(self, question: str) -> bool:
        q = question.lower()
        return any(keyword in q for keyword in self.web_keywords)

    def is_medical_question(self, question: str) -> bool:
        q = question.lower()
        return any(keyword in q for keyword in self.medical_keywords)

    def extract_city(self, question: str) -> str:
        """
        Very simple city extractor for demo purposes.
        """
        q = question.lower()

        patterns = [
            r"météo à ([a-zA-ZÀ-ÿ\- ]+)",
            r"meteo a ([a-zA-ZÀ-ÿ\- ]+)",
            r"temps à ([a-zA-ZÀ-ÿ\- ]+)",
            r"temps de ([a-zA-ZÀ-ÿ\- ]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                return match.group(1).strip().title()

        return "Paris"

    def route(self, question: str) -> dict:
        """
        Route the question to the appropriate subsystem.
        Returns a dict with:
        - type
        - answer
        - sources
        """
        if self.is_calculation(question):
            answer = calculator_tool(question)
            return {
                "type": "calculator",
                "answer": answer,
                "sources": []
            }

        if self.is_weather_question(question):
            city = self.extract_city(question)
            answer = weather_tool(city)
            return {
                "type": "weather",
                "answer": answer,
                "sources": []
            }

        if self.is_web_search_question(question):
            answer = web_search_tool(question)
            return {
                "type": "web_search",
                "answer": answer,
                "sources": []
            }

        if self.is_medical_question(question):
            history = self.memory.get_history()
            chain_input = {
                "question": question,
                "history": history
            }

            answer = self.rag_chain.invoke(chain_input)

            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)

            return {
                "type": "rag",
                "answer": answer,
                "sources": ["RAG medical documents"]
            }

        fallback_prompt = (
            "Réponds STRICTEMENT en français de manière claire et concise.\n"
            f"Question utilisateur : {question}"
        )
        answer = self.chat_llm.invoke(fallback_prompt).content

        self.memory.add_user_message(question)
        self.memory.add_ai_message(answer)

        return {
            "type": "chat",
            "answer": answer,
            "sources": []
        }
    
if __name__ == "__main__":
    router = AssistantRouter()

    print("\n" + "-" * 50)
    print("HYBRID ASSISTANT IS ONLINE")
    print("Type 'quit' to exit")
    print("-" * 50 + "\n")

    while True:
        user_input = input("Your Question: ").strip()

        if user_input.lower() in ["quit", "exit", "stop", "quitter"]:
            print("Closing the assistant. Stay safe!")
            break

        if not user_input:
            continue

        print("\nAnalyzing and routing your question...\n")
        result = router.route(user_input)

        print("--- Route Selected ---")
        print(result["type"])

        print("\n--- Assistant's Response ---")
        print(result["answer"])

        if result["sources"]:
            print("\n--- Sources ---")
            for source in result["sources"]:
                print(f"- {source}")

        print("\n" + "-" * 50 + "\n")