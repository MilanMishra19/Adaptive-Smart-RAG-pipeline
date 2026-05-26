import argparse
import json
import os
import sys
import time
import numpy as np
from sentence_transformers import SentenceTransformer
#Args
def parse_args():
    p = argparse.ArgumentParser(description="Build all RAG pipeline indexes.")
    p.add_argument("--papers",       type=int,  default=10_000,
                   help="Number of arXiv papers to load (default: 10000)")
    p.add_argument("--skip-data",    action="store_true",
                   help="Skip dataset download; load CHUNKS_PATH directly")
    p.add_argument("--chunks-file",  type=str,  default=None,
                   help="Path to a pre-built chunks JSON file")
    p.add_argument("--only",         type=str,  default=None,
                   choices=["embeddings", "bm25", "qdrant"],
                   help="Rebuild only one index (requires chunks + embeddings to exist)")
    p.add_argument("--batch-size",   type=int,  default=512,
                   help="Qdrant upsert batch size (default: 512)")
    return p.parse_args()

#Helper functions
def step(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def check_env():
    missing = [k for k in ("GROQ_API_KEY",) if not os.getenv(k)]
    if missing:
        print(f"[WARN] Missing env vars: {', '.join(missing)}")
        print("       The pipeline will fail at query time without these.")
        print("       Set them in a .env file or your shell before running queries.\n")


#Loading and chunking
def build_chunks(args, config) -> list[dict]:
    # --chunks-file or --skip-data: load existing JSON
    source = args.chunks_file or (config.CHUNKS_PATH if args.skip_data else None)
    if source:
        if not os.path.exists(source):
            sys.exit(f"[ERROR] Chunks file not found: {source}")
        step(f"Loading chunks from {source}")
        with open(source) as f:
            chunks = json.load(f)
        print(f"[INFO] Loaded {len(chunks)} chunks")
        return chunks

    step(f"Downloading arXiv dataset ({args.papers} papers)")
    from rag_pipeline.data.load_data import load_arxiv_subset, process_dataset
    data   = load_arxiv_subset(n=args.papers)
    chunks = process_dataset(data)

    out = config.CHUNKS_PATH
    with open(out, "w") as f:
        json.dump(chunks, f)
    print(f"[INFO] Saved {len(chunks)} chunks → {out}")
    return chunks


#Embedding
def build_embeddings(chunks: list[dict], config) -> "np.ndarray":
    out = config.EMBEDDINGS_PATH
    if os.path.exists(out):
        print(f"[INFO] Embeddings already exist at {out} — skipping.")
        print("       Delete the file and re-run if you want to rebuild.")
        return np.load(out)

    step(f"Encoding {len(chunks)} chunks with {config.BI_ENCODER_MODEL}")
    model  = SentenceTransformer(config.BI_ENCODER_MODEL)
    texts  = [c["text"] for c in chunks]

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"[INFO] Encoded in {time.time()-t0:.1f}s  shape={embeddings.shape}")
    np.save(out, embeddings)
    print(f"[INFO] Saved embeddings → {out}")
    return embeddings

#BM25 index
def build_bm25(chunks: list[dict], config):
    out = config.BM25_INDEX_PATH
    if os.path.exists(out):
        print(f"[INFO] BM25 index already exists at {out} — skipping.")
        print("       Delete the folder and re-run if you want to rebuild.")
        return

    step(f"Building BM25 index → {out}")
    from rag_pipeline.retrieval.bm25 import create_bm25
    create_bm25(chunks)

#Qdrant Vector DB
def build_qdrant(chunks: list[dict], embeddings, config):
    step(f"Building Qdrant collection '{config.COLLECTION_NAME}'")
    print("[INFO] Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()           # connection check
    except Exception as e:
        sys.exit(f"[ERROR] Cannot connect to Qdrant: {e}\n"
                 f"        Start it with: docker run -p 6333:6333 qdrant/qdrant")

    from rag_pipeline.retrieval.semantic import create_qdrant_index
    create_qdrant_index(chunks, embeddings, client, batch_size=args.batch_size)

def main():
    global args
    args = parse_args()
    check_env()
    sys.path.insert(0, os.path.dirname(__file__))
    import rag_pipeline.config as config
    if args.only:
        import numpy as np
        with open(config.CHUNKS_PATH) as f:
            chunks = json.load(f)

        if args.only == "embeddings":
            # Force rebuild by removing existing file
            if os.path.exists(config.EMBEDDINGS_PATH):
                os.remove(config.EMBEDDINGS_PATH)
            build_embeddings(chunks, config)

        elif args.only == "bm25":
            import shutil
            if os.path.exists(config.BM25_INDEX_PATH):
                shutil.rmtree(config.BM25_INDEX_PATH)
            build_bm25(chunks, config)

        elif args.only == "qdrant":
            embeddings = np.load(config.EMBEDDINGS_PATH)
            build_qdrant(chunks, embeddings, config)

        print("\n[DONE]")
        return
    chunks     = build_chunks(args, config)
    embeddings = build_embeddings(chunks, config)
    build_bm25(chunks, config)
    build_qdrant(chunks, embeddings, config)

    print(f"""
{'='*60}
  Setup complete!

  Files created:
    {config.CHUNKS_PATH}
    {config.EMBEDDINGS_PATH}
    {config.BM25_INDEX_PATH}/
    Qdrant collection: '{config.COLLECTION_NAME}'

  Next steps:
    1. Set GROQ_API_KEY in your environment
    2. from rag_pipeline.app import Pipeline
       pipe = Pipeline()
       print(pipe.query("what is swarm optimization"))
{'='*60}
""")


if __name__ == "__main__":
    main()