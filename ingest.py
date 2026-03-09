import os
import faiss
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.stores import InMemoryStore
from langchain_community.docstore.in_memory import InMemoryDocstore

# 🚨 THE PATH 2 UPGRADE: Local Open-Source Embeddings 🚨
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(dotenv_path=".env") 

def build_advanced_retriever(file_path):
    print(f"Loading document: {file_path}")
    loader = Docx2txtLoader(file_path)
    docs = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 1. Initialize the local HuggingFace model
    print("Spinning up local embedding model (first run takes a moment to download)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Set FAISS to exactly 384 dimensions (required for all-MiniLM)
    index = faiss.IndexFlatL2(384) 
    
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print("Chunking and embedding document... This might take a few seconds.")
    retriever.add_documents(docs)
    print("Done! The Beast has ingested the contract locally.")
    
    return retriever

if __name__ == "__main__":
    file_path = "../data/safe_note.docx" 
    retriever = build_advanced_retriever(file_path)
    
    query = "What is the definition of a Liquidity Event?"
    print(f"\nSearching for: '{query}'")
    
    retrieved_docs = retriever.invoke(query)
    
    print("\n--- RETRIEVED PARENT CONTEXT ---")
    if retrieved_docs:
        print(retrieved_docs[0].page_content[:500] + "...\n[TRUNCATED]")
    else:
        print("No documents retrieved.")