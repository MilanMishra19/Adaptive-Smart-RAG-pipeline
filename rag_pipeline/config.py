import os
from groq import Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY env var is not set")
    return Groq(api_key=GROQ_API_KEY)

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./notebook/bm25_index_10k")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./notebook/qdrant_database_10k")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "./all_chunks_slim.json")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./chunk_embeddings_10k (1).npy")

BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
LLM_MODEL = "llama-3.1-8b-instant"

TOP_K_RETRIEVE = 50
TOP_N_RERANK = 5
FETCH_K = 100

VECTOR_SIZE = 384
COLLECTION_NAME = "papers"
CONFIDENCE_THRESHOLD = 0.45