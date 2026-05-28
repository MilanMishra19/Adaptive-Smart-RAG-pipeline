---
title: Adaptive RAG Backend
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---
# Adaptive Hybrid RAG Pipeline

A retrieval-augmented generation system over 443k chunks from 10,000 arXiv papers. Combines BM25, dense vector search, and a cross-encoder reranker with LLM-based query routing, query rewriting, and retrieval grading.

## How it works

A query goes through four stages:

1. **Route** — an LLM classifier picks the retrieval strategy (`bm25`, `dense`, `hybrid`, or `hybrid_rerank`) based on query type
2. **Retrieve** — hybrid search with query expansion via RRF fusion, with up to 2 automatic retries if retrieval confidence is low. On each retry the query is rewritten using a different strategy before trying again
3. **Rerank** — a cross-encoder scores the top candidates; skipped if pre-rerank confidence already clears the threshold
4. **Generate** — Groq LLaMA produces a grounded answer from the retrieved chunks

## Benchmark (50 queries, corpus-grounded)

| Metric | Score |
|---|---|
| Recall@50 pre-rerank | 0.880 |
| Recall@5 post-rerank (exact) | 0.820 |
| Recall@5 post-rerank (paper) | 0.900 |
| MRR exact | 0.663 |
| MRR paper | 0.830 |
| Reranker lift @5 | +0.040 |
| Generation eval (10 queries) | 10/10 relevant |

## Stack

| Component | Library |
|---|---|
| Sparse retrieval | `bm25s` |
| Dense retrieval | `all-MiniLM-L6-v2` + Qdrant |
| Fusion | Reciprocal Rank Fusion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| LLM (routing / generation / grading) | Groq `llama-3.1-8b-instant` |

## Setup

**Prerequisites:** Python 3.11+, Docker (for Qdrant)

```bash
git clone https://github.com/MilanMishra19/adaptive-rag
cd adaptive-rag
pip install -r requirements.txt
```

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

```bash
# Set env vars
export GROQ_API_KEY=your_key_here
```

```bash
# Build all indexes (downloads dataset, encodes chunks, builds BM25 + Qdrant)
python setup.py
```

This takes a while on first run — it encodes 443k chunks. Subsequent runs skip already-built indexes automatically. If you have a laptop lacking a GPU, I suggest using Google Colab with runtime set to T4 GPU. Download the embeddings in the form of .npy as well as .json

**Custom dataset size:**
```bash
python setup.py --papers 5000
```

**Bring your own chunks:**
```bash
python setup.py --chunks-file ./my_chunks.json
```

**Rebuild a single index:**
```bash
python setup.py --only bm25
python setup.py --only qdrant
python setup.py --only embeddings
```

## Usage

```python
from rag_pipeline.app import Pipeline

pipe = Pipeline()  # loads all indexes once
result = pipe.query("any query you would like..")

print(result["answer"])
print(result["strategy"])   # which retrieval strategy was used
print(result["confidence"]) # float in [0, 1]
print(result["sources"])    # retrieved chunks with scores
```

## Project structure

```
rag_pipeline/
├── data/
│   └── load_data.py        # dataset download, cleaning, chunking
├── retrieval/
│   ├── bm25.py             # BM25 index build + search
│   ├── semantic.py         # Qdrant index build + dense search
│   └── hybrid.py           # RRF fusion, query expansion
├── reranking/
│   └── rerank.py           # cross-encoder rerank, confidence scoring, retry loop
├── routing/
│   └── router.py           # LLM-based routing 
├── generation/
│   └── generate.py         # to generate queries grounded to the corpus
├── eval_ragas/
│   ├── metrics.py          # recall, MRR, grading
│   └── benchmark.py        # benchmark builder
├── config.py               # all paths and model names, env-configurable, make necessary changes here only
└── app.py                  # Pipeline class — combines the module
setup.py                    # first-time index builder for adding your own dataset etc.
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required |
| `HF_TOKEN` | — | Only needed for gated HuggingFace datasets |
| `BM25_INDEX_PATH` | `./bm25_index_10k` | Where to save/load the BM25 index |
| `QDRANT_PATH` | `localhost:6333` | Qdrant connection |
| `CHUNKS_PATH` | `./all_chunks_slim.json` | Chunks JSON |
| `EMBEDDINGS_PATH` | `./chunk_embeddings.npy` | Embeddings array |
