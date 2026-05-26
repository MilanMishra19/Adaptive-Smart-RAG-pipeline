
# Adaptive Hybrid RAG System

A modular, research-oriented Adaptive Retrieval-Augmented Generation (RAG) system built from first principles using:
- Sparse Retrieval (BM25)
- Dense Semantic Retrieval
- Hybrid Fusion (RRF)
- Cross-Encoder Reranking
- Retrieval Grading
- Retry Logic
- LLM-based Query Routing

This project focuses on **adaptive retrieval orchestration** rather than building a simple chatbot wrapper.

---

# Project Vision

Traditional RAG systems use:
- one retriever
- one vector database
- static retrieval logic

This project instead explores:

```text
adaptive retrieval policy optimization
```

where the system dynamically:
- selects retrieval strategies
- grades retrieval quality
- retries when retrieval confidence is low
- reranks results
- routes queries intelligently using an LLM

The core goal is to create a retrieval system that behaves more like an intelligent retrieval engine rather than a fixed pipeline.

---

# High-Level Architecture

```text
                    ┌────────────────────┐
                    │   User Query       │
                    └─────────┬──────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   LLM Query Router     │
                 │ (Policy Selection)     │
                 └─────────┬──────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     BM25 Search      Dense Search      Hybrid Search
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                ┌──────────────────────┐
                │  Hybrid Fusion (RRF) │
                └─────────┬────────────┘
                          ▼
                ┌──────────────────────┐
                │ Cross Encoder        │
                │ Reranking            │
                └─────────┬────────────┘
                          ▼
                ┌──────────────────────┐
                │ Retrieval Grading    │
                └─────────┬────────────┘
                          ▼
                Low Confidence?
                          │
                    Yes ──┴──► Retry Logic
                          │
                         No
                          ▼
                ┌──────────────────────┐
                │ Final Retrieved      │
                │ Context              │
                └──────────────────────┘
```

---

# Core Features

## 1. Scientific Paper Dataset Processing

The system is designed primarily for:
- Mathematics,Space research papers
- scientific literature
- long-form technical documents

Current dataset source:
- Hugging Face arXiv dataset

---

## 2. Advanced Chunking Pipeline

Documents undergo:
- text cleaning
- section-aware splitting
- recursive semantic chunking

### Chunking Strategy
- Section-aware regex splitting
- RecursiveCharacterTextSplitter
- Metadata enrichment
- Chunk hashing

### Metadata Stored

Each chunk contains:
- chunk_id
- paper_id
- section_title
- chunk_index
- token_estimate
- abstract preview

---

# Retrieval System

## Sparse Retrieval (BM25)

Implements:
- lexical retrieval
- exact terminology matching
- acronym-sensitive retrieval

Best for:
- exact paper titles
- terminology-heavy queries
- benchmark names
- abbreviations

---

## Dense Retrieval

Uses transformer embeddings for semantic similarity search.

### Embedding Models Explored
- BGE-small
- SPECTER
- MiniLM

Dense retrieval is best for:
- semantic reasoning
- conceptual similarity
- paraphrased queries

---

## Vector Database

Vector storage is handled using Qdrant.

Used for:
- scalable vector search
- persistent embedding storage
- similarity retrieval

---

# Hybrid Retrieval

The system combines:
- sparse retrieval
- dense retrieval

using:

## Reciprocal Rank Fusion (RRF)

```math
RRF(d)=\sum_{r\in R}\frac{1}{k+r(d)}
```

### Why RRF?

RRF:
- is robust
- handles score distribution mismatch
- works well across heterogeneous retrievers
- is widely used in production retrieval systems

---

# Cross-Encoder Reranking

After retrieval, candidate chunks are reranked using a cross encoder.

### Reranker Model

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Why Reranking?

Retrieval retrieves:
- approximate candidates

Reranking improves:
- precision
- ranking quality
- semantic relevance

This creates a:

```text
multi-stage retrieval pipeline
```

---

# Adaptive Retrieval Logic

## Retrieval Grading

Retrieved contexts are scored using:
- retrieval confidence
- hybrid score quality
- ranking consistency

Low-confidence retrievals trigger retries.

---

## Retry Logic

The system supports adaptive retries:
- alternate retrieval strategies
- retrieval fallback mechanisms
- reranking retries

Example:

```text
Hybrid Retrieval Failed
        ↓
Retry using Dense-Heavy Retrieval
```

---

# LLM Query Routing

The retrieval strategy is dynamically selected using an LLM router.

### Router Model

```text
Qwen2.5-3B-Instruct
```

The router chooses:
- retrieval strategy
- top_k
- reranking policy

Example router output:

```json
{
  "strategy": "hybrid",
  "top_k": 10,
  "rerank": true
}
```

---

# Evaluation System

The project evaluates retrieval quality using classical Information Retrieval metrics.

## Metrics Implemented

### Recall@K

Measures whether relevant chunks appear in the top K retrieved results.

### Mean Reciprocal Rank (MRR)

```math
MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}
```

Measures ranking quality.

### Hit Rate

Checks whether at least one relevant chunk was retrieved.

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Notebook Environment | Jupyter |
| Sparse Retrieval | rank-bm25 |
| Embeddings | SentenceTransformers |
| Vector Database | Qdrant |
| Reranking | CrossEncoder |
| LLM Routing | Groq + Llama |
| Chunking | LangChain Text Splitters |
| Dataset | Hugging Face arXiv |

---
# Planned Future Work

## Generation Pipeline

Add:
- final answer synthesis
- context-aware generation
- grounded response generation

---

## Advanced Evaluation

Future additions:
- RAGAS
- Context Precision
- Faithfulness
- Answer Relevancy

---

## Graph Retrieval

Potential future integration:
- citation graphs
- author relationships
- multi-hop retrieval

---

## Structured Retrieval

Potential additions:
- PostgreSQL
- metadata retrieval
- benchmark querying

---

## Retrieval Failure Analysis

Analyze:
- lexical misses
- semantic misses
- reranker failures
- routing failures

---

# Design Philosophy

This project intentionally avoids:
- premature agent systems
- orchestration complexity
- framework overdependence

Instead, it focuses on:
- retrieval quality
- adaptive retrieval policies
- modular experimentation
- systems-level retrieval engineering

---

# Key Insight

This is NOT:

```text
"just another RAG chatbot"
```

The system is designed as:

```text
an adaptive retrieval orchestration framework
```

where retrieval itself becomes:
- dynamic
- policy-driven
- self-correcting
- evaluation-aware

---

# Example Query Flow

```text
User Query:
"methods that repair retrieval failures"

        ↓

LLM Router
→ chooses Dense Retrieval + Reranking

        ↓

Dense Retrieval
→ retrieves semantic candidates

        ↓

Reranker
→ improves ranking precision

        ↓

Retrieval Grader
→ evaluates retrieval confidence

        ↓

Retry Logic (if needed)

        ↓

Final Context Returned
```

---

# Repository Structure (Planned)

```text
adaptive-rag/
│___data/
|    └── data.py
|
├── retrieval/
│   ├── bm25.py
│   ├── dense.py
│   ├── hybrid.py
│   └── reranker.py
│
├── routing/
│   ├── llm_router.py
│   └── retry_logic.py
│
├── evaluation/
│   ├── metrics.py
│   └── benchmark.py
│
├── storage/
│   └── qdrant_store.py
│
├── notebooks/
│   └── experimentation.ipynb
│
└── README.md
```
---
#Setup
python -m rag_pipeline.py
#Initializing local server for Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
#Verify it is running
docker ps
#Test modular structure
python
>>>from rag_pipeline.app import Pipeline
>>>pipe = Pipeline()
>>>result = pipe.query("your query here")
>>>print(result)
```

