
import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Paths — resolved relative to this file's location, works on any machine
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FOLDER = os.path.join(BASE_DIR, "data", "raw")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def extract_title(content: str, filename: str) -> str:
    """
    Attempts to extract the disease title from the file content.
    Looks for a line "Titre : <name>" in the text.
    If nothing is found, uses the filename without its extension.
    """
    match = re.search(r"Titre\s*:\s*(.+)", content, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return os.path.splitext(filename)[0].replace("-", " ").capitalize()


def load_documents(data_folder: str) -> list:
    """
    Loads all TXT and PDF files from the given folder.
    Returns a list of LangChain documents with enriched metadata.
    """
    documents = []

    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            file_type = "pdf"
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            file_type = "txt"
        else:
            continue

        print(f"Loading {filename}...")
        docs = loader.load()

        # Extract title from the full content of the loaded document
        full_content = " ".join([d.page_content for d in docs])
        title = extract_title(full_content, filename)

        # Attach metadata to each page/document
        for doc in docs:
            doc.metadata["source_file"] = filename
            doc.metadata["file_type"] = file_type
            doc.metadata["disease_title"] = title

        documents.extend(docs)

    print(f"\nTotal: {len(documents)} document(s) loaded")
    return documents


def split_into_chunks(documents: list) -> list:
    """
    Splits documents into fixed-size chunks with overlap.
    Metadata from the parent document is preserved on each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks created")
    return chunks


def embed_and_save(chunks: list, index_path: str):
    """
    Computes embeddings for each chunk and saves the FAISS index.
    """
    print(f"\nComputing embeddings with model '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_path)
    print(f"FAISS index saved to: {index_path}")


def preview_metadata(chunks: list, n: int = 3):
    """
    Prints metadata for the first n chunks for verification.
    """
    print(f"\nMetadata preview for the first {n} chunks:")
    print("-" * 50)
    for i, chunk in enumerate(chunks[:n]):
        print(f"Chunk {i + 1}")
        print(f"  source_file    : {chunk.metadata.get('source_file')}")
        print(f"  file_type      : {chunk.metadata.get('file_type')}")
        print(f"  disease_title  : {chunk.metadata.get('disease_title')}")
        print(f"  content (start): {chunk.page_content[:80]}...")
        print()


if __name__ == "__main__":
    print("Starting ingestion pipeline\n")

    documents = load_documents(DATA_FOLDER)
    chunks = split_into_chunks(documents)
    preview_metadata(chunks)
    embed_and_save(chunks, INDEX_PATH)

    print("\nIngestion complete.")
