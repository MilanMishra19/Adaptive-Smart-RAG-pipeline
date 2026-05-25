# rag_pipeline/retrieval/hybrid.py

from groq import Groq
from bm25s import BM25
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rag_pipeline.retrieval.bm25 import bm25_search
from rag_pipeline.retrieval.semantic import semantic_search
from rag_pipeline.config import TOP_K_RETRIEVE, FETCH_K, LLM_MODEL

def reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 30,
) -> list[dict]:
    """
    Fuse any number of ranked result lists using RRF.
    Accepts *result_lists so it works for 2-way (bm25 + dense)
    and N-way (multi-query) fusion with the same function.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            cid = str(result["chunk_id"])
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_map[cid] = result

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    output = []
    for cid in sorted_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = scores[cid]
        output.append(chunk)

    return output

def hybrid_search(
    query: str,
    retriever: BM25,
    client: QdrantClient,
    bi_encoder: SentenceTransformer,
    all_chunks: list[dict],
    top_k: int = TOP_K_RETRIEVE,
    fetch_k: int = FETCH_K,
) -> list[dict]:
    bm25_results    = bm25_search(query, retriever, all_chunks, top_k=fetch_k)
    semantic_results = semantic_search(query, client, bi_encoder, top_k=fetch_k)
    fused = reciprocal_rank_fusion(bm25_results, semantic_results)
    return fused[:top_k]

def expand_query(
    query: str,
    groq_client: Groq,
    n: int = 2,
) -> list[str]:
    """Returns [original] + n expansion variants."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} alternative search queries for a scientific paper retrieval system.\n"
                f"Original: {query}\n\n"
                f"Rules:\n"
                f"- Do NOT repeat or rephrase the original query\n"
                f"- Use different vocabulary — avoid repeating key terms from the original\n"
                f"- One must use broader conceptual language\n"
                f"- One must use related technical synonyms or adjacent concepts\n\n"
                f"Return ONLY the {n} new queries, one per line, no numbering, no original."
            )
        }]
    )
    lines = response.choices[0].message.content.strip().split("\n")
    variants = [
        l.strip() for l in lines
        if l.strip() and l.strip().lower() != query.lower()
    ]
    return [query] + variants[:n]

def hybrid_search_expanded(
    query: str,
    retriever: BM25,
    client: QdrantClient,
    bi_encoder: SentenceTransformer,
    all_chunks: list[dict],
    groq_client: Groq,
    top_k: int = TOP_K_RETRIEVE,
    fetch_k: int = FETCH_K,
) -> list[dict]:
    """
    Expand query into variants, run hybrid_search per variant,
    then do a final RRF pass across ALL per-query result lists.
    This ensures a chunk that ranks well across multiple query
    variants gets properly boosted — not just whoever came first.
    """
    queries = expand_query(query, groq_client)

    per_query_results = [
        hybrid_search(q, retriever, client, bi_encoder, all_chunks,
                      top_k=fetch_k, fetch_k=fetch_k)
        for q in queries
    ]
    fused = reciprocal_rank_fusion(*per_query_results)
    return fused[:top_k]