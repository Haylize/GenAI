# GenAI: End-to-End Retrieval-Augmented Generation (RAG) System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This repository hosts a production-grade **End-to-End GenAI Pipeline**. The project focuses on building a scalable RAG (Retrieval-Augmented Generation) system that allows users to interact with private datasets using Large Language Models (LLMs).

Unlike simple notebooks, this project implements a modular architecture including automated data ingestion, vector indexing, and a deployment-ready API/UI.

## Key Features
* **Modular Data Pipeline:** Automated ingestion and cleaning of unstructured data (PDFs, Markdown, Docx).
* **Advanced RAG Strategy:** Implementation of semantic search using Vector Databases (ChromaDB/Pinecone).
* **Context-Aware Chat:** Integrated memory for multi-turn conversations.
* **Production Ready:** Logging, exception handling, and containerization using Docker.
* **User Interface:** Interactive dashboard built with Streamlit for real-time testing.

## Project Structure
The project is organized into modular components to ensure maintainability:

```
.
├── .github/workflows      # CI/CD pipelines (GitHub Actions)
├── assets/                # Architecture diagrams and UI screenshots
├── configs/               # YAML files for model parameters & paths
├── data/                  # Local storage for raw and processed documents
├── notebook/              # Research and experimentation (Jupyter)
├── src/                   # Core Source Code
│   ├── components/        # Ingestion, Transformation, Embedding, Generation
│   ├── pipeline/          # Training and Prediction (Inference) pipelines
│   ├── constants/         # Fixed variables and file paths
│   ├── entity/            # Data classes for configuration
│   ├── logger/            # Custom logging module
│   └── exception/         # Custom exception handling
├── static/                # CSS/JS for the UI
├── templates/             # HTML templates (if using Flask/FastAPI)
├── app.py                 # Main application entry point (Streamlit/FastAPI)
├── setup.py               # Package installation script
├── Dockerfile             # Containerization instructions
└── requirements.txt       # Project dependencies
```