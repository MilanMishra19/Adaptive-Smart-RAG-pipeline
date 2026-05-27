# tests/test_metrics.py
from rag_pipeline.eval_ragas.metrics import recall_at_k_exact
def test_recall_at_k_exact_hit():
    results = [{"chunk_id": "abc"}, {"chunk_id": "xyz"}]
    assert recall_at_k_exact(results, "abc", k=5) == 1.0

def test_recall_at_k_exact_miss():
    results = [{"chunk_id": "abc"}]
    assert recall_at_k_exact(results, "zzz", k=5) == 0.0
