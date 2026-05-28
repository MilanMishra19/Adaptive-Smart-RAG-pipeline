"""
pipeline.py — single entry point for the full RAG pipeline.

Usage:
    from rag_pipeline.pipeline import Pipeline

    pipe = Pipeline()                          # loads all indexes once
    result = pipe.query("what is swarm optimization")
    print(result["answer"])
"""

from __future__ import annotations

import json
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
import bm25s

from rag_pipeline.config import (
    GROQ_API_KEY,
    BM25_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    BI_ENCODER_MODEL,
    CROSS_ENCODER_MODEL,
    TOP_N_RERANK,
    CONFIDENCE_THRESHOLD,
    HF_REPO_ID,
    HF_TOKEN
)


class Pipeline:
    """Loads every index and model once at construction; exposes .query()."""
    def ensure_artifacts(self):
        from huggingface_hub import snapshot_download
        import os
        needs_download = any([
            not os.path.exists(CHUNKS_PATH),
            not os.path.exists(BM25_INDEX_PATH),
            not os.path.exists(EMBEDDINGS_PATH),
        ])
        if needs_download:
            print("[INFO] Artifacts missing, downloading from hf data repository")
            snapshot_download(
                repo_id = HF_REPO_ID,
                repo_type="dataset",
                local_dir="./data",
                token = HF_TOKEN,
            )
            print("[INFO] Download complete")
        else:
            print("[INFO] Artifacts alr present")
    def __init__(self) -> None:
        # LLM client
        print("[INFO] Pipeline init started")
        self._ensure_artifacts()
        print("[INFO] Artifacts ready")
        self.groq_client = Groq(api_key=GROQ_API_KEY)

        # Retrieval models
        self.bi_encoder   = SentenceTransformer(BI_ENCODER_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        # Indexes
        self.retriever = bm25s.BM25.load(BM25_INDEX_PATH, load_corpus=True)
        self.qdrant    = QdrantClient(host="localhost",port=6333)

        # Corpus + embeddings
        with open(CHUNKS_PATH) as f:
            self.all_chunks: list[dict] = json.load(f)
        self.embeddings = np.load(EMBEDDINGS_PATH)

        # Push loaded objects into module singletons so sub-modules
        # never re-load a model during a request.
        self._inject_singletons()

    # ------------------------------------------------------------------
    # Singleton injection
    # ------------------------------------------------------------------

    def _inject_singletons(self) -> None:
        import rag_pipeline.retrieval.bm25     as _bm25
        import rag_pipeline.retrieval.semantic as _sem
        import rag_pipeline.reranking.rerank   as _rerank

        _bm25._retriever  = self.retriever
        _bm25._all_chunks = self.all_chunks

        _sem._bi_encoder  = self.bi_encoder
        _sem._qdrant      = self.qdrant
        _sem._all_chunks  = self.all_chunks

        _rerank._cross_encoder = self.cross_encoder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        *,
        use_router: bool = True,
        max_retries: int = 2,
    ) -> dict:
        """Run the full pipeline for a single query.

        Args:
            query:       Natural language question.
            use_router:  If True, LLM classifies the retrieval strategy.
                         If False, defaults to hybrid_rerank for everything.
            max_retries: Query-rewrite retry attempts on retrieval failure.

        Returns a dict with keys:
            query, final_query, answer, strategy, top_k, rerank,
            confidence, grade, retrieval_failed, attempts, sources
        """
        from rag_pipeline.routing.router       import llm_route
        from rag_pipeline.reranking.rerank     import retrieve_with_retry
        from rag_pipeline.generation.generate import generate_answer

        # 1. Route -------------------------------------------------------
        if use_router:
            route     = llm_route(query, groq_client=self.groq_client)
            strategy  = route.get("strategy", "hybrid_rerank")
            top_k     = route.get("top_k", TOP_N_RERANK)
            do_rerank = route.get("rerank", True)
        else:
            strategy, top_k, do_rerank = "hybrid_rerank", TOP_N_RERANK, True

        # 2. Retrieve (with retry + query rewriting) ----------------------
        retrieval  = retrieve_with_retry(
            query,
            max_retries=max_retries,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            groq_client=self.groq_client,
        )
        results    = retrieval["results"]
        final_q    = retrieval["final_query"]
        confidence = retrieval["confidence"]
        grade      = retrieval["grade"]
        failed     = retrieval["retrieval_failed"]

        # 3. Generate ----------------------------------------------------
        if failed or not results:
            answer = retrieval.get(
                "fallback_message",
                "I couldn't find relevant information for your query.",
            )
        else:
            answer = generate_answer(
                final_q,
                results,
                top_k=top_k,
                groq_client=self.groq_client,
            )

        return {
            "query":            query,
            "final_query":      final_q,
            "answer":           answer,
            "strategy":         strategy,
            "top_k":            top_k,
            "rerank":           do_rerank,
            "confidence":       confidence,
            "grade":            grade,
            "retrieval_failed": failed,
            "attempts":         retrieval.get("attempts", []),
            "sources":          results,
        }