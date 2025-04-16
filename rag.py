from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def load_and_process_documents():
    """Load and split PDF documents from docs/ directory"""
    loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

def initialize_vector_store():
    """Create FAISS vector store with embeddings"""
    documents = load_and_process_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embeddings)

# Initialize vector store at startup
vector_db = initialize_vector_store()