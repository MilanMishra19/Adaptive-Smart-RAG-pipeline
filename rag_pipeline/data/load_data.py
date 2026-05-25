#/rag_pipeline/data.py
import re
import json
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset
import numpy as np
from config import CHUNKS_PATH,EMBEDDINGS_PATH
def clean_text(text):
    text = re.sub(r"@xcite", "", text)       
    text = re.sub(r"@xmath\d+", "[MATH]", text)  
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_arxiv_subset(n: int = 10000):
    dataset = load_dataset("ccdv/arxiv-summarization")
    return dataset["train"].select(range(n))


def process_dataset(data) -> list[dict]:
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for i, paper in enumerate(data):
        print(f"[INFO] Processing document {i}...")
        try:
            article  = clean_text(paper.get("article", ""))
            abstract = clean_text(paper.get("abstract", ""))

            # Abstract chunks
            for chunk_idx, chunk in enumerate(splitter.split_text(abstract)):
                chunk = chunk.strip()
                if len(chunk) < 50:
                    continue
                all_chunks.append({
                    "chunk_id":      hashlib.md5(f"{i}_abstract_{chunk_idx}".encode()).hexdigest(),
                    "paper_id":      i,
                    "section_title": "ABSTRACT",
                    "chunk_index":   chunk_idx,
                    "text":          f"{abstract[:200]} {chunk}",
                    "abstract":      abstract[:300],
                    "token_estimate": len(chunk.split()),
                })

            # Article chunks
            for chunk_idx, chunk in enumerate(splitter.split_text(article)):
                chunk = chunk.strip()
                if len(chunk) < 50:
                    continue
                all_chunks.append({
                    "chunk_id":      hashlib.md5(f"{i}_article_{chunk_idx}".encode()).hexdigest(),
                    "paper_id":      i,
                    "section_title": "ARTICLE",
                    "chunk_index":   chunk_idx,
                    "text":          chunk,
                    "abstract":      abstract[:300],
                    "token_estimate": len(chunk.split()),
                })

        except Exception as e:
            print(f"[ERROR] Failed processing document {i}: {e}")

    print(f"\n[INFO] Created {len(all_chunks)} chunks from {len(data)} papers")
    return all_chunks


def retrieve_data() -> tuple[list[dict], np.ndarray]:
    with open(CHUNKS_PATH) as f:
        all_chunks = json.load(f)

    chunk_embeddings = np.load(EMBEDDINGS_PATH)

    print(f"[INFO] Loaded {len(all_chunks)} chunks and embeddings shape {chunk_embeddings.shape}")
    return all_chunks, chunk_embeddings