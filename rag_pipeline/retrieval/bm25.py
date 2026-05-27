# rag_pipeline/retrieval/bm25.py

import bm25s
from rag_pipeline.config import BM25_INDEX_PATH, TOP_K_RETRIEVE

_retriever = None
_all_chunks: list[dict] =[]
def create_bm25(all_chunks: list[dict]) -> None:
    corpus_texts = [c["text"] for c in all_chunks]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus_texts))
    retriever.save(BM25_INDEX_PATH)
    print(f"[INFO] BM25 index saved to {BM25_INDEX_PATH}")

def load_bm25() -> bm25s.BM25:
    retriever = bm25s.BM25.load(BM25_INDEX_PATH, load_corpus=False)
    print(f"[INFO] BM25 loaded from {BM25_INDEX_PATH}")
    return retriever

def bm25_search(
    query: str,
    retriever: bm25s.BM25,
    all_chunks: list[dict],
    top_k: int = TOP_K_RETRIEVE,
) -> list[dict]:
    tokenized = bm25s.tokenize(query)
    results, scores = retriever.retrieve(tokenized, k=top_k)
    output = []
    for idx, score in zip(results[0], scores[0]):
        chunk = all_chunks[int(idx)].copy()   
        chunk["bm25_score"] = float(score)
        chunk["chunk_id"] = str(chunk["chunk_id"])
        output.append(chunk)
    return output