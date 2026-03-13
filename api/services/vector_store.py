import os
import json
import re
from typing import List, Dict, Any, Optional
from collections import Counter

from qdrant_client.models import Filter, FieldCondition, MatchText


class VectorStore:
    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def add_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ):
        from qdrant_client.models import PointStruct

        if ids is None:
            ids = [str(i) for i in range(len(vectors))]

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(vectors))
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)

    def _build_filter(
        self,
        document_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Optional[Filter]:
        conditions = []
        if document_name:
            conditions.append(
                FieldCondition(key="document_name", match=MatchText(text=document_name))
            )
        if document_id:
            conditions.append(
                FieldCondition(key="document_id", match=MatchText(text=document_id))
            )

        if conditions:
            return Filter(must=conditions)
        return None

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        query_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
                "text": result.payload.get("text", ""),
                "document_id": result.payload.get("document_id", ""),
                "document_name": result.payload.get("document_name", ""),
                "chunk_index": result.payload.get("chunk_index", 0),
            }
            for result in results
        ]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 5,
        query_filter: Optional[Dict] = None,
        alpha: float = 0.7,
    ) -> List[Dict[str, Any]]:
        vector_results = self.search(
            query_vector=query_vector,
            top_k=top_k * 3,
            query_filter=query_filter,
        )

        query_terms = self._tokenize(query_text)
        keyword_results = self._keyword_search(
            query_terms=query_terms,
            results=vector_results,
            top_k=top_k * 3,
        )

        combined = {}
        for r in vector_results:
            combined[r["id"]] = {
                **r,
                "vector_score": r["score"],
                "keyword_score": 0.0,
            }
        for r in keyword_results:
            if r["id"] in combined:
                combined[r["id"]]["keyword_score"] = r["score"]
            else:
                combined[r["id"]] = {
                    **r,
                    "vector_score": 0.0,
                    "keyword_score": r["score"],
                }

        for r in combined.values():
            r["score"] = alpha * r["vector_score"] + (1 - alpha) * r["keyword_score"]

        sorted_results = sorted(
            combined.values(), key=lambda x: x["score"], reverse=True
        )
        return sorted_results[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _compute_bm25_score(self, query_terms: List[str], text: str) -> float:
        doc_terms = self._tokenize(text)
        doc_freq = Counter(doc_terms)
        query_freq = Counter(query_terms)

        score = 0.0
        for term, qf in query_freq.items():
            if term in doc_freq:
                tf = doc_freq[term]
                score += (tf * (1.5 + 1)) / (tf + 1.5)

        return score

    def _keyword_search(
        self,
        query_terms: List[str],
        results: List[Dict],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        scored = []
        for r in results:
            text = r.get("text", "")
            score = self._compute_bm25_score(query_terms, text)
            if score > 0:
                scored.append({**r, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)
