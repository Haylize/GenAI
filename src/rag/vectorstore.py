import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION & PATHS ---
# Calculate the project root directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_local_retriever():
    """
    Initializes embeddings and loads the local FAISS index.
    Returns a retriever object configured for similarity search.
    """
    print(f"--- Loading FAISS Index from {INDEX_PATH} ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Check if the index exists before trying to load it
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at: {INDEX_PATH}")

    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Return retriever with top-k results parameter (k=3)
    return vectorstore.as_retriever(search_kwargs={"k": 3})