# rag_pipeline/retrieval/semantic.py

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from rag_pipeline.config import BI_ENCODER_MODEL,TOP_K_RETRIEVE,COLLECTION_NAME,VECTOR_SIZE    

def load_qdrant() -> QdrantClient:
    client = QdrantClient(host="localhost",port=6333)
    print("[INFO] Qdrant loaded")
    return client


def load_bi_encoder() -> SentenceTransformer:
    model = SentenceTransformer(BI_ENCODER_MODEL)
    print(f"[INFO] Bi-encoder loaded: {BI_ENCODER_MODEL}")
    return model

def create_qdrant_index(
    all_chunks: list[dict],
    chunk_embeddings: np.ndarray,
    client: QdrantClient,
    batch_size: int = 512,
) -> None:
    """Build Qdrant collection from scratch. Safe to re-run — drops first if exists."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Dropped existing collection '{COLLECTION_NAME}'")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[INFO] Created collection '{COLLECTION_NAME}'")

    for start in range(0, len(all_chunks), batch_size):
        end = min(start + batch_size, len(all_chunks))
        points = [
            PointStruct(
                id=i,
                vector=chunk_embeddings[i].tolist(),
                payload={
                    "chunk_id": all_chunks[i]["chunk_id"],
                    "paper_id": all_chunks[i]["paper_id"],
                    "section":  all_chunks[i]["section_title"],
                    "text":     all_chunks[i]["text"],
                },
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

        if start % 50000 == 0:
            print(f"[INFO] Indexed {start}/{len(all_chunks)}")

    print(f"[INFO] Qdrant indexing complete — {len(all_chunks)} points")

def semantic_search(
    query: str,
    client: QdrantClient,
    bi_encoder: SentenceTransformer,
    top_k: int = TOP_K_RETRIEVE,
) -> list[dict]:
    """
    Dense vector search over Qdrant.
    client and bi_encoder passed in explicitly — no globals, no model loads here.
    """
    query_emb = bi_encoder.encode([query])[0].tolist()

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb,
        limit=top_k,
    ).points

    results = []
    for hit in hits:
        r = dict(hit.payload)
        r["section_title"] = r.pop("section", "UNKNOWN")
        r["chunk_id"]      = str(r.get("chunk_id", ""))
        r["dense_score"]   = float(hit.score)
        results.append(r)

    return results