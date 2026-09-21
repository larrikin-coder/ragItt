import os
from dotenv import load_dotenv


load_dotenv()

PINECONE_API = os.getenv("PINECONE_API")
PINECONE_ENV = os.getenv("PINECONE_ENV","us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX","rag-index")

GROQ_API = os.getenv("GROQ_API")
TAVILY_DIR = os.getenv("TAVILY_DIR")
DOC_SOURCE_DIR = os.getenv("DOC_SOURCE_DIR","data")

EMBED_MODEL = os.getenv("EMBED_MODEL")

