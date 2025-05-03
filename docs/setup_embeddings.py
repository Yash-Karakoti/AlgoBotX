# setup_embeddings.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load your PDF
loader = PyPDFLoader("PYTEAL.pdf")  # Replace with your file name
documents = loader.load()

# Step 2: Create Embeddings using Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Step 3: Create a FAISS vectorstore
vectorstore = FAISS.from_documents(documents, embeddings)

# Step 4: Save the vectorstore
vectorstore.save_local("my_vectorstore")

print("PDF processed and vectorstore saved!")

#Yet to be completed (on process)