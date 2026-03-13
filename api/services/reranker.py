import os
from typing import List, Tuple
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[dict], top_k: int = 5) -> List[dict]:
        if not results:
            return []

        doc_texts = [(query, r["text"]) for r in results]
        scores = self.model.predict(doc_texts)

        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

        reranked = []
        for i, (result, score) in enumerate(ranked[:top_k]):
            result["rerank_score"] = float(score)
            result["original_rank"] = i + 1
            reranked.append(result)

        return reranked
