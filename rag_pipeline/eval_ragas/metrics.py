"""
evaluation/metrics.py
 
Pure metric functions (recall, MRR) plus LLM-based grading and the two
full eval runners. All functions that touch retrieval/generation/routing
accept callables so this module stays decoupled from the rest of the
package — the Pipeline wires everything together in pipeline.py.
"""
 
import re
import numpy as np
from collections import defaultdict
from groq import Groq
from rag_pipeline.config import LLM_MODEL
 
 
# ---------------------------------------------------------------------------
# Pure metrics — no I/O, no LLM
# ---------------------------------------------------------------------------
 
def recall_at_k_exact(results: list, gt_chunk_id: str, k: int = 5) -> int:
    """1 if the exact ground-truth chunk appears in the top-k results."""
    return int(any(str(r["chunk_id"]) == gt_chunk_id for r in results[:k]))
 
 
def recall_at_k_paper(results: list, gt_paper_id: int, k: int = 5) -> int:
    """1 if any chunk from the ground-truth paper appears in the top-k results."""
    return int(any(r.get("paper_id") == gt_paper_id for r in results[:k]))
 
 
def mrr_exact(results: list, gt_chunk_id: str) -> float:
    """Mean Reciprocal Rank — strict: must match the exact chunk."""
    for rank, r in enumerate(results, start=1):
        if str(r["chunk_id"]) == gt_chunk_id:
            return 1 / rank
    return 0.0
 
 
def mrr_paper(results: list, gt_paper_id: int) -> float:
    """Mean Reciprocal Rank — soft: any chunk from the right paper counts."""
    for rank, r in enumerate(results, start=1):
        if r.get("paper_id") == gt_paper_id:
            return 1 / rank
    return 0.0
 
 
# ---------------------------------------------------------------------------
# LLM-based grading
# ---------------------------------------------------------------------------
 
def grading(query: str, rank_results: list, confidence: float, groq_client: Groq) -> tuple[str, str]:
    """
    Use the LLM to judge whether the retrieved chunks actually answer the query.
 
    Returns:
        (grade, reason) where grade is "relevant" | "partial" | "irrelevant".
    """
    if not rank_results:
        return "irrelevant", "no results found"
 
    top_chunks = "\n\n--\n\n".join(r["text"][:300] for r in rank_results[:3])
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""You are a retrieval quality grader.
Query: {query}
Retrieved chunks:
{top_chunks}
Grade the retrieval. Respond in this exact format:
GRADE: <relevant|partial|irrelevant>
REASON: <one sentence>"""
        }]
    )
    text = response.choices[0].message.content.strip()
    grade_match  = re.search(r"GRADE:\s*(relevant|partial|irrelevant)", text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", text)
    grade  = grade_match.group(1).lower()  if grade_match  else "irrelevant"
    reason = reason_match.group(1).strip() if reason_match else "unknown"
    return grade, reason
 
 
# ---------------------------------------------------------------------------
# Full eval runners — accept callables so they stay decoupled
# ---------------------------------------------------------------------------
 
def run_grounded_eval(
    benchmark: list,
    hybrid_search_expanded_fn,
    rerank_fn,
    top_k_retrieve: int = 50,
    top_n_rerank: int = 5,
) -> dict:
    """
    Offline retrieval benchmark over a grounded query set.
 
    Args:
        benchmark:               List of benchmark dicts from build_grounded_benchmark().
        hybrid_search_expanded_fn: Callable(query, top_k) -> list of chunk dicts.
        rerank_fn:               Callable(query, chunks, top_n) -> list of chunk dicts.
        top_k_retrieve:          How many raw candidates to retrieve before reranking.
        top_n_rerank:            How many results to keep after reranking.
 
    Returns:
        metrics dict (keys → list of per-query scores).
    """
    metrics: defaultdict[str, list[float]] = defaultdict(list) #fixed type issues
    failed  = []
 
    for item in benchmark:
        query    = item["query"]
        gt_chunk = item["ground_truth_chunk_id"]
        gt_paper = item["ground_truth_paper_id"]
        section  = item["source_section"]
 
        try:
            raw     = hybrid_search_expanded_fn(query, top_k=top_k_retrieve)
            results = rerank_fn(query, raw, top_n=top_n_rerank)
 
            r5_raw    = recall_at_k_exact(raw[:top_n_rerank], gt_chunk, k=top_n_rerank)
            r5_rerank = recall_at_k_exact(results,            gt_chunk, k=top_n_rerank)
            r5_exact  = recall_at_k_exact(results, gt_chunk, k=top_n_rerank)
            r5_pre    = recall_at_k_exact(raw,     gt_chunk, k=top_k_retrieve)
            r5_paper  = recall_at_k_paper(results, gt_paper, k=top_n_rerank)
            r50_paper = recall_at_k_paper(raw,     gt_paper, k=top_k_retrieve)
 
            metrics["recall@5_raw"].append(r5_raw)
            metrics["recall@5_reranked"].append(r5_rerank)
            metrics["recall@5_exact"].append(r5_exact)
            metrics["recall@5_paper"].append(r5_paper)
            metrics[f"recall@5_exact_{section}"].append(r5_exact)
            metrics[f"recall@5_paper_{section}"].append(r5_paper)
            metrics["recall@50_pre_exact"].append(r5_pre)
            metrics["recall@50_pre_paper"].append(r50_paper)
            metrics["mrr_exact"].append(mrr_exact(results, gt_chunk))
            metrics["mrr_paper"].append(mrr_paper(results, gt_paper))
 
            icon = "A" if r5_exact else ("B" if r5_paper else ("C" if r5_pre else "F"))
            print(f"{icon} [{section}] exact={r5_exact} paper={r5_paper} | {query[:60]}")
 
        except Exception as e:
            failed.append(query)
            print(f"[ERROR] {query[:60]} — {e}")
 
    n_queries = len(benchmark) - len(failed)
    print(f"\n--- results ({n_queries} queries) ---")
    print(f"recall@50 pre-rerank  (exact) : {np.mean(metrics['recall@50_pre_exact']):.3f}")
    print(f"recall@5  post-rerank (exact) : {np.mean(metrics['recall@5_exact']):.3f}")
    print(f"recall@5  post-rerank (paper) : {np.mean(metrics['recall@5_paper']):.3f}")
    print(f"mrr exact                     : {np.mean(metrics['mrr_exact']):.3f}")
    print(f"mrr paper                     : {np.mean(metrics['mrr_paper']):.3f}")
 
    for section in ["ABSTRACT", "ARTICLE"]:
        k = f"recall@5_exact_{section}"
        if metrics[k]:
            print(f"\n{section.lower()} ({len(metrics[k])} chunks)")
            print(f"  exact : {np.mean(metrics[k]):.3f}")
            print(f"  paper : {np.mean(metrics[f'recall@5_paper_{section}']):.3f}")
 
    true_lift = np.mean(metrics["recall@5_reranked"]) - np.mean(metrics["recall@5_raw"])
    print(f"\ntrue reranker lift (reranked@5 vs raw@5) : {true_lift:+.3f}")
 
    if failed:
        print(f"\nfailed on {len(failed)} queries: {failed}")
 
    return dict(metrics)
 
 
def run_generation_eval(
    benchmark: list,
    llm_route_fn,
    bm25_search_fn,
    semantic_search_fn,
    hybrid_search_expanded_fn,
    retrieve_with_retry_fn,
    rerank_fn,
    generate_answer_fn,
    grading_fn,
    top_k_retrieve: int = 50,
    top_n_rerank: int = 5,
    n: int = 10,
) -> list:
    """
    End-to-end generation quality eval over a random sample of the benchmark.
 
    All retrieval/generation callables are injected so this function stays
    decoupled from the rest of the package.
 
    Args:
        benchmark:                  List of benchmark dicts.
        llm_route_fn:               Callable(query) -> {"strategy": str, ...}
        bm25_search_fn:             Callable(query, top_k) -> list
        semantic_search_fn:         Callable(query, top_k) -> list
        hybrid_search_expanded_fn:  Callable(query, top_k) -> list
        retrieve_with_retry_fn:     Callable(query) -> dict with "results" key
        rerank_fn:                  Callable(query, chunks, top_n) -> list
        generate_answer_fn:         Callable(query, results) -> str
        grading_fn:                 Callable(query, results, confidence) -> (grade, reason)
        top_k_retrieve:             Candidates before reranking.
        top_n_rerank:               Kept after reranking.
        n:                          Number of benchmark items to sample.
 
    Returns:
        List of per-query result dicts.
    """
    import random
    results_log = []
    sample = random.sample(benchmark, min(n, len(benchmark)))
 
    for item in sample:
        query    = item["query"]
        gt_chunk = item["ground_truth_chunk_id"]
        gt_paper = item["ground_truth_paper_id"]
 
        route = llm_route_fn(query)
        strategy = route["strategy"]
 
        if strategy == "bm25":
            raw = bm25_search_fn(query, top_k=top_k_retrieve)
        elif strategy == "dense":
            raw = semantic_search_fn(query, top_k=top_k_retrieve)
        elif strategy == "hybrid":
            raw = hybrid_search_expanded_fn(query, top_k=top_k_retrieve)
        else:  # hybrid_rerank — use retry mechanism
            retry_result = retrieve_with_retry_fn(query)
            raw = retry_result.get("results", [])
 
        reranked   = rerank_fn(query, raw, top_n=top_n_rerank)
        answer     = generate_answer_fn(query, reranked)
        confidence = reranked[0].get("rrf_score", 1.0) if reranked else 0.0
        grade, reason = grading_fn(query, reranked, confidence)
 
        results_log.append({
            "query":    query,
            "strategy": strategy,
            "grade":    grade,
            "reason":   reason,
            "answer":   answer,
            "gt_chunk": gt_chunk,
            "gt_paper": gt_paper,
        })
 
        icon = "✅" if grade == "relevant" else ("⚠️" if grade == "partial" else "❌")
        print(f"{icon} [{grade}] {query[:60]}")
        print(f"   → {answer[:120]}")
        print(f"   reason: {reason}\n")
 
    grades = [r["grade"] for r in results_log]
    print("--- generation eval summary ---")
    print(f"relevant  : {grades.count('relevant')}/{len(grades)}")
    print(f"partial   : {grades.count('partial')}/{len(grades)}")
    print(f"irrelevant: {grades.count('irrelevant')}/{len(grades)}")
 
    return results_log