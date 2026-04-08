from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Charger le PDF
print("Chargement du PDF...")
loader = PyPDFLoader("data/test_fiche.pdf")
documents = loader.load()
print(f"{len(documents)} page(s) chargée(s)")

# 2. Découper en morceaux
print("Découpage en chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"{len(chunks)} chunks créés")

# 3. Vectoriser et sauvegarder
print("Vectorisation...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")
print(" Index FAISS sauvegardé dans faiss_index/")
