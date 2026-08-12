import os
import shutil
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Shared embedding model singleton to avoid reloading on every call
_embedding_cache = {}

def _get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in _embedding_cache:
        _embedding_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    return _embedding_cache[model_name]

def is_vectorstore_populated(persist_directory="db_folder"):
    """Check if the persist directory exists and is not empty."""
    if os.path.exists(persist_directory):
        return len(os.listdir(persist_directory)) > 0
    return False

def build_vectorstore(file_path: str, persist_directory="db_folder", chunk_size=700, chunk_overlap=150):
    """Load a PDF, split it into chunks, and store it in Chroma DB.
    
    Uses ChromaDB's client API to clear existing data instead of deleting
    locked files on disk, which avoids WinError 32 on Windows.
    """
    # If an existing database exists, use ChromaDB client to clear it safely
    if os.path.exists(persist_directory):
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            # Delete all existing collections
            for col in client.list_collections():
                client.delete_collection(col.name)
            del client  # Release the client before proceeding
        except Exception as e:
            print(f"Warning: Could not clear existing DB via API ({e}), attempting file delete...")
            try:
                shutil.rmtree(persist_directory)
            except Exception as e2:
                print(f"Warning: Could not delete db_folder ({e2}), will overwrite.")

    loader = PyPDFLoader(file_path)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(document)

    embedding_obj = _get_embeddings()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_obj,
        persist_directory=persist_directory
    )

    return len(chunks)

def load_vectorstore(persist_directory="db_folder"):
    """Legacy wrapper for backward compatibility."""
    file_path = "mypdf.pdf"
    if os.path.exists(file_path):
        build_vectorstore(file_path, persist_directory)
    return None

def load_retriever(persist_directory="db_folder", search_k=4, search_type="mmr"):
    """Load the retriever from the persisted vector store.
    
    Supports 'similarity' and 'mmr' (Maximal Marginal Relevance) for high-accuracy retrieval.
    """
    embedding_obj = _get_embeddings()
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_obj
    )
    if search_type == "mmr":
        return vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": search_k,
                "fetch_k": max(20, search_k * 4),
                "lambda_mult": 0.7
            }
        )
    return vector_db.as_retriever(search_type="similarity", search_kwargs={"k": search_k})



def clear_vectorstore(persist_directory="db_folder"):
    """Clear the database safely using ChromaDB API first, then file delete as fallback."""
    if os.path.exists(persist_directory):
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            for col in client.list_collections():
                client.delete_collection(col.name)
            del client
        except Exception:
            pass
        try:
            shutil.rmtree(persist_directory)
            return True
        except Exception as e:
            print(f"Error removing directory {persist_directory}: {e}")
            return False
    return False

