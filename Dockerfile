FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_pipeline/ ./rag_pipeline/
COPY backend/ ./backend/

ENV BM25_INDEX_PATH=/app/data/bm25_index_10k
ENV QDRANT_PATH=/app/data/qdrant_database_10k
ENV CHUNKS_PATH=/app/data/all_chunks_slim.json
ENV EMBEDDINGS_PATH=/app/data/chunk_embeddings.npy

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]