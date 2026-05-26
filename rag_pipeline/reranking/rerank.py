#/rag_pipeline/reranking/rerank.py
import re
import numpy as np
from sentence_transformers import CrossEncoder
 
from rag_pipeline.config import (
    CROSS_ENCODER_MODEL,
    TOP_N_RERANK,
    TOP_K_RETRIEVE,
    CONFIDENCE_THRESHOLD,
)
from rag_pipeline.retrieval.hybrid import hybrid_search,hybrid_search_expanded
from rag_pipeline.retrieval.bm25 import bm25_search
from rag_pipeline.retrieval.semantic import semantic_search
from rag_pipeline.eval_ragas.metrics import grading
 
_cross_encoder: CrossEncoder | None = None
 
 
def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder
 
def rerank(query: str, top_k_chunks: list[dict], top_n: int = TOP_N_RERANK) -> list[dict]:
    """Score every (query, chunk) pair with the cross-encoder and return the
    top-n chunks sorted by descending cross-encoder score."""
    encoder = get_cross_encoder()
    pairs = [(query, c["text"]) for c in top_k_chunks]
    scores = encoder.predict(pairs)
    ranked = sorted(zip(scores, top_k_chunks), key=lambda x: x[0], reverse=True)
    results = []
    for score, chunk in ranked[:top_n]:
        chunk = chunk.copy()
        chunk["rerank_score"] = float(score)
        results.append(chunk)
    return results
 
def confidence_score(rank_results: list[dict]) -> tuple[float, list[float]]:
    """Return a scalar confidence in [0, 1] and the raw scores list.
 
    Combines top score, average score, and score drop-off between rank-1
    and rank-2.  Uses rerank_score when available, falls back to rrf_score.
    """
    if not rank_results:
        return 0.0, []
 
    scores = [r.get("rerank_score", r.get("rrf_score", 0.0)) for r in rank_results]
    min_s, max_s = min(scores), max(scores)
 
    if max_s > min_s:
        norm = [(s - min_s) / (max_s - min_s) for s in scores]
    else:
        norm = [1.0] * len(scores)
 
    top_score = norm[0]
    avg_score = float(np.mean(norm))
    dropoff = norm[0] - norm[1] if len(norm) > 1 else 0.0
 
    confidence = 0.5 * top_score + 0.3 * avg_score + 0.2 * min(dropoff * 2, 1.0)
    return round(float(confidence), 4), scores
 
def retrieve_and_grade(
    query: str,
    top_k: int = TOP_N_RERANK,
    fetch_k: int = TOP_K_RETRIEVE,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    groq_client=None,
) -> dict:
    """Single-shot retrieve + conditional rerank + grade.
 
    Skips the cross-encoder when pre-rerank confidence already meets the
    threshold (saves latency on easy queries).
    """
    if groq_client is None:
        from rag_pipeline.config import get_groq_client
        groq_client = get_groq_client()
 
    raw = hybrid_search_expanded(query, top_k=fetch_k)
    pre_confidence, _ = confidence_score(raw[:top_k])
 
    if pre_confidence >= confidence_threshold:
        results = raw[:top_k]
        confidence = pre_confidence
    else:
        results = rerank(query, raw, top_n=top_k)
        confidence, _ = confidence_score(results)
 
    grade, reason = grading(query, results, confidence, groq_client=groq_client)
    failed = grade == "irrelevant" or confidence < confidence_threshold
 
    return {
        "query": query,
        "results": results if not failed else [],
        "confidence": confidence,
        "grade": grade,
        "reason": reason,
        "retrieval_failed": failed,
        "fallback_message": "No relevant information found for this query." if failed else None,
    }
 
_REWRITE_STRATEGIES = {
    1: "Rephrase the query using more technical/academic language",
    2: "Break the query into its core concept only. Make it short, concise and to the point",
    3: "Rewrite the query using different synonyms or related words",
}
 
 
def rewrite_query(
    original_query: str,
    grade: str,
    reason: str,
    attempt: int = 1,
    groq_client=None,
) -> str:
    """Ask the LLM to rewrite a failing query using a strategy keyed by attempt number."""
    if groq_client is None:
        from rag_pipeline.config import get_groq_client
        groq_client = get_groq_client()
 
    strat = _REWRITE_STRATEGIES.get(attempt, _REWRITE_STRATEGIES[1])
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                f"You are a query rewriting assistant for a scientific paper retrieval system.\n\n"
                f"Original query: {original_query}\n"
                f"Retrieval grade: {grade}\n"
                f"Reason it failed: {reason}\n"
                f"Rewriting strategy: {strat}\n\n"
                f"Respond with ONLY the rewritten query, nothing else."
            ),
        }],
    )
    return response.choices[0].message.content.strip()
 
def _build_retry_strategies() -> list:
    """Build strategy callables at call time, after Pipeline._inject_singletons()
    has populated the module-level vars in each retrieval module."""
    import rag_pipeline.retrieval.bm25     as _bm25
    import rag_pipeline.retrieval.semantic as _sem
    return [
        ("hybrid",   lambda q: hybrid_search(
                        q,
                        retriever=_bm25._retriever,
                        client=_sem._qdrant,
                        bi_encoder=_sem._bi_encoder,
                        all_chunks=_bm25._all_chunks,
                        top_k=TOP_K_RETRIEVE,
                    )),
        ("bm25",     lambda q: bm25_search(
                        q,
                        retriever=_bm25._retriever,
                        all_chunks=_bm25._all_chunks,
                        top_k=TOP_K_RETRIEVE,
                    )),
        ("semantic", lambda q: semantic_search(
                        q,
                        client=_sem._qdrant,
                        bi_encoder=_sem._bi_encoder,
                        all_chunks=_sem._all_chunks,
                        top_k=TOP_K_RETRIEVE,
                    )),
    ]
 
def retrieve_with_retry(
    query: str,
    max_retries: int = 2,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    groq_client=None,
) -> dict:
    """Try up to max_retries+1 times, cycling through retrieval strategies and
    rewriting the query on each failure before the next attempt."""
    if groq_client is None:
        from rag_pipeline.config import get_groq_client
        groq_client = get_groq_client()
 
    attempt_log = []
    current_query = query
 
    retry_strategies = _build_retry_strategies()
    for attempt in range(max_retries + 1):
        strategy_name, search_fn = retry_strategies[min(attempt, len(retry_strategies) - 1)]
 
        raw_results = search_fn(current_query)
        pre_confidence, _ = confidence_score(raw_results[:TOP_N_RERANK])
 
        if pre_confidence >= confidence_threshold:
            results = raw_results[:TOP_N_RERANK]
            confidence = pre_confidence
        else:
            results = rerank(current_query, raw_results, top_n=TOP_N_RERANK)
            confidence, _ = confidence_score(results)
 
        grade, reason = grading(current_query, results, confidence, groq_client=groq_client)
        failed = grade == "irrelevant" or confidence < confidence_threshold
 
        attempt_log.append({
            "attempt": attempt,
            "query": current_query,
            "strategy": strategy_name,
            "confidence": confidence,
            "grade": grade,
            "reason": reason,
            "failed": failed,
        })
 
        if not failed:
            return {
                "query": query,
                "final_query": current_query,
                "results": results,
                "confidence": confidence,
                "grade": grade,
                "reason": reason,
                "retrieval_failed": False,
                "attempts": attempt_log,
            }
 
        if attempt >= max_retries:
            break
 
        current_query = rewrite_query(
            query, grade, reason,
            attempt=attempt + 1,
            groq_client=groq_client,
        )
 
    return {
        "query": query,
        "final_query": current_query,
        "results": [],
        "confidence": 0.0,
        "grade": "irrelevant",
        "reason": "all retry attempts failed",
        "retrieval_failed": True,
        "fallback_message": "I couldn't find relevant information after multiple attempts.",
        "attempts": attempt_log,
    }