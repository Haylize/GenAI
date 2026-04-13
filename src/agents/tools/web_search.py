import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_ollama import ChatOllama

load_dotenv()

API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOllama(model="mistral", temperature=0)

def web_search_tool(query: str) -> str:
    if not API_KEY:
        return "Clé API web search manquante. Vérifiez votre fichier .env."

    try:
        client = TavilyClient(api_key=API_KEY)

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=3
        )

        results = response.get("results", [])
        if not results:
            return "Aucun résultat trouvé."

        raw_results = []
        for i, result in enumerate(results, start=1):
            title = result.get("title", "Sans titre")
            url = result.get("url", "Pas d'URL")
            content = result.get("content", "Pas de résumé")

            raw_results.append(
                f"{i}. {title}\nURL: {url}\nRésumé: {content}"
            )

        context = "\n\n".join(raw_results)

        prompt = f"""
Tu es un assistant utile.
À partir des résultats web suivants, réponds en français de manière claire, concise et structurée.
Fais une petite synthèse, puis liste les sources à la fin.

Question utilisateur :
{query}

Résultats web :
{context}
"""

        final_answer = llm.invoke(prompt).content
        return final_answer

    except Exception as e:
        return f"Erreur lors de la recherche web : {str(e)}"