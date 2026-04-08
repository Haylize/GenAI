# Assistant RAG Multi-Compétences

## Stack technique
- LangChain + ChromaDB/FAISS
- LLM : OpenAI GPT-4o / Mistral
- Interface : Chainlit

## Installation
```bash
git clone ...
cd mon-assistant-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # remplir les clés
```

## Lancer l'app
```bash
chainlit run src/app.py
```
