import json
import random

from groq import Groq
from rag_pipeline.config import LLM_MODEL


def is_informative_chunk(chunk: dict, min_words: int = 30, max_math_ratio: float = 0.15) -> bool:
    """Return True if the chunk has enough real text (not math-heavy, not too short)."""
    text = chunk["text"] if isinstance(chunk, dict) else chunk
    tokens = text.split()
    if len(tokens) < min_words:
        return False
    math_tokens = sum(1 for t in tokens if "[MATH]" in t)
    return math_tokens / len(tokens) <= max_math_ratio


def generate_query_from_chunk(chunk_text: str, groq_client: Groq) -> str:
    """Ask the LLM to write a natural retrieval query for the given passage."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=80,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": f"""You are helping build a retrieval benchmark for scientific papers.

Given this passage:
\"\"\"{chunk_text[:450]}\"\"\"

Write ONE natural search query that a researcher would type to find information like this.
- Use plain language, not a copy of the passage
- Do NOT use quotes or bullet points
- Respond with ONLY the query, nothing else"""
        }]
    )
    return response.choices[0].message.content.strip().strip('"').strip("'")


def build_grounded_benchmark(
    all_chunks: list,
    groq_client: Groq,
    n: int = 50,
    seed: int = 42,
) -> list:
    """
    Sample informative chunks from the corpus and generate one query per chunk.

    Args:
        all_chunks:  Full list of chunk dicts loaded from all_chunks_slim.json.
        groq_client: Initialised Groq() client (injected by Pipeline or __main__).
        n:           Number of benchmark items to generate.
        seed:        Random seed for reproducibility.

    Returns:
        List of benchmark dicts, each with query + ground-truth ids.
    """
    random.seed(seed)
    pool = [c for c in all_chunks if is_informative_chunk(c)]
    print(f"usable chunks: {len(pool):,} / {len(all_chunks):,}")

    abstract_pool = [c for c in pool if c.get("section_title") == "ABSTRACT"]
    article_pool  = [c for c in pool if c.get("section_title") == "ARTICLE"]
    n_abstract = min(n // 2, len(abstract_pool))
    n_article  = min(n - n_abstract, len(article_pool))

    sampled = random.sample(abstract_pool, n_abstract) + random.sample(article_pool, n_article)
    random.shuffle(sampled)

    print(f"generating {len(sampled)} queries...")
    benchmark = []
    for idx, chunk in enumerate(sampled):
        try:
            query = generate_query_from_chunk(chunk["text"], groq_client)
            benchmark.append({
                "id": idx,
                "query": query,
                "ground_truth_chunk_id": str(chunk["chunk_id"]),
                "ground_truth_paper_id": chunk["paper_id"],
                "source_section": chunk.get("section_title", "UNKNOWN"),
                "source_text_preview": chunk["text"][:200],
            })
            print(f"  [{idx+1}/{len(sampled)}] {query[:70]}")
        except Exception as e:
            print(f"  [{idx+1}/{len(sampled)}] failed: {e}")

    return benchmark


if __name__ == "__main__":
    import os
    import json as _json
    from groq import Groq as _Groq
    from rag_pipeline.config import CHUNKS_PATH, GROQ_API_KEY

    with open(CHUNKS_PATH) as f:
        all_chunks = _json.load(f)

    client = _Groq(api_key=GROQ_API_KEY)
    grounded_benchmark = build_grounded_benchmark(all_chunks, groq_client=client, n=50)

    with open("grounded_benchmark.json", "w") as f:
        _json.dump(grounded_benchmark, f, indent=2)
    print("saved to grounded_benchmark.json")