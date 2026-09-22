import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


pc = Pinecone(api_key=PINECONE_API)

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def get_retriever():
    """Initiates pinecone and retrieves vector store embeddings"""
    
    if PINECONE_INDEX not in pc.list_indexes().names():
        print("Creating the index in the vector database")
        pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
        )
        print("Pinecone Index created")
    vector_store = PineconeVectorStore(embedding=embedding, index=PINECONE_INDEX)
    return vector_store.as_retriever()

def add_documents(text_content:str):
    """
    Adds a single text document to the pinecone vector database.
    Splits the text into chunks before embeddings and upserting.
    """    
    if not text_content:
        raise ValueError("Document content cannot be empty.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    documents = text_splitter.create_documents([text_content])
    print("Splitting document into chunk for indexing...")
    vector_store = PineconeVectorStore(embedding=embedding, index=PINECONE_INDEX)
    vector_store.add_documents(documents)
    print("Successfully added chunks to pinecone vectorstore")
