import os
from dotenv import load_dotenv


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV","us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX","rag-index")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DOC_SOURCE_DIR = os.getenv("DOC_SOURCE_DIR","data")

EMBED_MODEL = os.getenv("EMBED_MODEL")

print("PINECONE_API:", PINECONE_API_KEY)
print("GROQ_API:", GROQ_API_KEY)
print("EMBED_MODEL:", EMBED_MODEL)
