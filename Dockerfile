FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_pipeline/ ./rag_pipeline/
COPY backend/ ./backend/

ENV PYTHONPATH=/app
ENV BM25_INDEX_PATH=/app/data/bm25_index_10k
ENV QDRANT_PATH=/app/data/qdrant_database_10k
ENV CHUNKS_PATH=/app/data/all_chunks_slim.json
ENV EMBEDDINGS_PATH=/app/data/chunk_embeddings.npy

EXPOSE 7860

CMD ["sh", "-c", "echo 'Starting...' && ls /app && echo 'Python path:' && python -c 'import sys; print(sys.path)' && echo 'Testing import...' && python -c 'from backend.main import app; print(\"Import OK\")' && uvicorn backend.main:app --host 0.0.0.0 --port 7860"]