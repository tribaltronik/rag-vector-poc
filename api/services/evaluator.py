from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class EvalResult:
    query: str
    expected_doc_id: str
    retrieved_doc_id: str
    relevant: bool
    precision_at_k: float
    recall: float


class RetrievalEvaluator:
    def __init__(self):
        self.results: List[EvalResult] = []

    def evaluate_query(
        self,
        query: str,
        expected_doc_id: str,
        retrieved_results: List[Dict[str, Any]],
        k: int = 5,
    ) -> EvalResult:
        retrieved_ids = [r.get("document_id", "") for r in retrieved_results[:k]]

        relevant = expected_doc_id in retrieved_ids

        retrieved_relevant = sum(1 for rid in retrieved_ids if rid == expected_doc_id)
        precision = retrieved_relevant / k if k > 0 else 0.0
        recall = 1.0 if relevant else 0.0

        result = EvalResult(
            query=query,
            expected_doc_id=expected_doc_id,
            retrieved_doc_id=retrieved_ids[0] if retrieved_ids else "",
            relevant=relevant,
            precision_at_k=precision,
            recall=recall,
        )

        self.results.append(result)
        return result

    def get_metrics(self) -> Dict[str, float]:
        if not self.results:
            return {
                "total_queries": 0,
                "precision_at_k_avg": 0.0,
                "recall_avg": 0.0,
                "hit_rate": 0.0,
            }

        total = len(self.results)
        precision_sum = sum(r.precision_at_k for r in self.results)
        recall_sum = sum(r.recall for r in self.results)
        hits = sum(1 for r in self.results if r.relevant)

        return {
            "total_queries": total,
            "precision_at_k_avg": precision_sum / total,
            "recall_avg": recall_sum / total,
            "hit_rate": hits / total,
        }

    def reset(self):
        self.results = []
