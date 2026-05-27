from groq import Groq
from rag_pipeline.config import  LLM_MODEL
def generate_answer(query: str, results: list, groq_client: Groq, top_k: int = 3) -> str:
    """
    Generate a grounded answer from top reranked chunks.
 
    Args:
        query:       The user's original query.
        results:     Reranked chunk list (dicts with at least a "text" key).
        groq_client: Initialised Groq() client (injected by Pipeline).
        top_k:       How many chunks to use as context.
 
    Returns:
        The LLM-generated answer string.
    """
    context = "\n\n---\n\n".join(r["text"][:400] for r in results[:top_k])
 
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""You are a scientific research assistant. Answer the query using only the provided context.
Context:
{context}
 
Query: {query}
 
Rules:
- Answer concisely in 2-3 sentences
- Only use information from the context
- If the context doesn't contain the answer, say so
Answer:"""
        }]
    )
    return response.choices[0].message.content.strip()