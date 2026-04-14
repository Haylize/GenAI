import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION & PATHS ---
# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_local_retriever():
    """
    Initialize embeddings, load the local FAISS index,
    and return a retriever configured for cleaner document retrieval.
    """
    print(f"--- Loading FAISS Index from {INDEX_PATH} ---")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at: {INDEX_PATH}")

    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Cleaner retrieval:
    # - k=2 limits noisy extra sources
    # - fetch_k=4 gives FAISS a slightly larger pool before selection
    # - MMR helps reduce redundant / less relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 4
        }
    )

    return retriever