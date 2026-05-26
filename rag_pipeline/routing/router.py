import re
import json


def llm_route(query: str, groq_client=None) -> dict:
    """Classify a query into a retrieval strategy config.

    Returns:
        {
            "strategy": "bm25" | "dense" | "hybrid" | "hybrid_rerank",
            "top_k": int,
            "rerank": bool,
        }
    """
    if groq_client is None:
        from rag_pipeline.config import get_groq_client
        groq_client = get_groq_client()

    prompt = f"""You are a deterministic retrieval policy classifier. Your sole function is to map an incoming query to the optimal retrieval configuration.

Available retrieval strategies:
1. bm25: Best for exact terminology, acronyms, specific codes, or exact paper/entity names.
2. dense: Best for semantic similarity, conceptual questions, and natural language phrasing.
3. hybrid: Best for mixed queries containing both specific keywords and broader concepts.
4. hybrid_rerank: Best for difficult, complex, or multi-hop questions requiring high precision.

[CRITICAL BEHAVIORAL CONSTRAINTS]
- Do NOT output any free-form reasoning, thinking blocks, or chain-of-thought.
- Do NOT output any explanations, introductions, or postscripts.
- You must behave strictly as a token-optimized classifier.
- Your output must be ONLY valid, minified JSON matching the schema below.

[OUTPUT SCHEMA]
{{
  "strategy": "bm25" | "dense" | "hybrid" | "hybrid_rerank",
  "top_k": integer,
  "rerank": boolean
}}

Input Query: {query}
JSON Output:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=60,    # routing JSON is tiny; cap it
        temperature=0.0,  # must be deterministic
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in LLM router response: {raw!r}")
        return json.loads(match.group())