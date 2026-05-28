from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams,PointStruct
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Local client
local = QdrantClient(path="./notebook/qdrant_database_10k")

# Cloud client
cloud = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

COLLECTION_NAME = "papers"  # change if yours is different

# Get collection info from local
local_info = local.get_collection(COLLECTION_NAME)
vector_size = local_info.config.params.vectors.size
print(f"[INFO] Collection: {COLLECTION_NAME}, vector size: {vector_size}")

# Create collection on cloud
if cloud.collection_exists(COLLECTION_NAME):
    cloud.delete_collection(COLLECTION_NAME)
cloud.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)
print("[INFO] Cloud collection created")

# Migrate in batches
batch_size = 100
offset = None
total = 0
total_points = local_info.points_count
print(f"[INFO] Total points to migrate: {total_points}")

while True:
    results, offset = local.scroll(
        collection_name=COLLECTION_NAME,
        limit=batch_size,
        offset=offset,
        with_vectors=True,
        with_payload=True,
    )
    
    if not results:
        break
    cloud.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=point.id,
            vector=point.vector,
            payload=point.payload,
        )
        for point in results
    ],
    )

    
    total += len(results)
    print(f"[INFO] Migrated {total} / {total_points} ({100*total//total_points}%)")
    
    if offset is None:
        break

print(f"[DONE] Migrated {total} points to Qdrant Cloud")