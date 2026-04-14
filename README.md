# Hybrid AI Assistant (RAG + Tools)

## Overview

This project is a **hybrid AI assistant** combining:

- **RAG (Retrieval-Augmented Generation)** for domain-specific knowledge (e.g. medical data)
- **LLM (Mistral via Ollama)** for natural language generation
- **External tools** (weather, web search, calculator)
- **Conversation memory** for contextual interactions

The assistant is able to:
- Answer questions using a **local knowledge base (FAISS)**
- Perform **real-time actions via tools**
- Maintain a **short-term conversation memory**

---

## Project Structure

```
GENAI/
│
├── data/
│ ├── raw/ # Raw data (documents, PDFs, etc.)
│ ├── processed/ # Cleaned / chunked data
│──faiss_index/ # Vector database (embeddings, not pushed)
│
├── src/
│ ├── agents/tools/
│ │ ├── calculator.py
│ │ ├── weather.py
│ │ └── web_search.py
│ │
│ ├── memory/
│ │ └── memory.py
│ │
│ ├── rag/
│ │ ├── ingest.py # Data ingestion & embedding
│ │ ├── vectorstore.py # FAISS loading
│ │ ├── qa_chain.py # RAG pipeline
│ │ └── retrieve.py # Interactive assistant
│ │
│ ├── router/
│ │ └── router.py # Routing logic (RAG vs tools)
│ │
│ └── app.py # Main entry point
│
├── .env # API keys (NOT pushed)
├── requirements.txt
└── README.md
```

## Features

### RAG System
- Uses **FAISS** for semantic search
- Retrieves relevant documents before answering
- Improves factual accuracy

### Tools Integration
- Weather API
- Web search
- Calculator

### Smart Routing
- Automatically decides:
  - Use RAG (knowledge base)
  - Use a tool (external API)
  - Use LLM directly

### Memory
- Stores last interactions
- Enables contextual conversations

## Installation

- Clone the repository
- Create virtual environment
- Install dependencies
- Setup environment variables
- Create a .env file at the root:
    - OPENWEATHER_API_KEY=your_key_here
    - SERPAPI_KEY=your_key_here
- Install Ollama
    - Download from: https://ollama.com
- Then install Mistral:
    - ollama pull mistral

## Usage

- Run the assistant: python src/rag/retrieve.py

- Ask any question:
  - Symptoms of flu?
  - Weather in Paris?
  - Search latest AI news

## How It Works

1 : User asks a question  
2 : Router decides:
  - Tool → execute API
  - RAG → retrieve documents
  - LLM → direct answer  
3 : Context is sent to Mistral  
4 : Response is generated  
5 : Memory is updated  

## Tech Stack

- Python
- FAISS (vector DB)
- Ollama (LLM local)
- Mistral model
- LangChain (optional depending on your impl)
- dotenv

## Known Limitations

- Requires local model (Ollama running)
- API keys must be configured manually
- RAG quality depends on data quality

## Next Improvements

- Improve routing logic (LLM-based router)
- Add more tools (finance, news, etc.)
- Deploy on cloud
- Add logging / monitoring

## Security
- API keys stored in .env
- .env excluded via .gitignore
