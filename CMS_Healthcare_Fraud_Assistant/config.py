import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "cms-healthcare-fraud")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "cms-fraud-v2")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "850"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "140"))
DENSE_CANDIDATES = int(os.getenv("DENSE_CANDIDATES", "14"))
LEXICAL_CANDIDATES = int(os.getenv("LEXICAL_CANDIDATES", "14"))
FINAL_CONTEXT_K = int(os.getenv("FINAL_CONTEXT_K", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "18000"))
FLASK_PORT = int(os.getenv("FLASK_PORT", "8080"))
