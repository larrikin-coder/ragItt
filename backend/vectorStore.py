import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PINECONE_API_KEY, PINECONE_INDEX, EMBED_MODEL

pc = Pinecone(api_key=PINECONE_API_KEY)
print("PINECONE_INDEX value:", PINECONE_INDEX)
print("Existing indexes:", pc.list_indexes().names())
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
    vector_store = PineconeVectorStore(embedding=embedding, index=pc.Index(PINECONE_INDEX))
    return vector_store.as_retriever()

def add_documents(text_content:str):
    """
    Adds a single text document to the pinecone vector database.
    Splits the text into chunks before embeddings and upserting.
    """    
    if not text_content:
        raise ValueError("Document content cannot be empty.")
    
    if PINECONE_INDEX not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX}' not found, creating...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Pinecone index created successfully")
    
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    documents = text_splitter.create_documents([text_content])
    print("Splitting document into chunk for indexing...")
    vector_store = PineconeVectorStore(embedding=embedding, index=pc.Index(PINECONE_INDEX))
    vector_store.add_documents(documents)
    print("Successfully added chunks to pinecone vectorstore")
