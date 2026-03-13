import os
import json
import httpx
from typing import List, Dict, Any, Optional

from qdrant_client.models import SparseVector, Filter, FieldCondition, MatchText


class VectorStore:
    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def add_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
    ):
        from qdrant_client.models import PointStruct, SparseVector

        if ids is None:
            ids = [str(i) for i in range(len(vectors))]

        points = []
        for i in range(len(vectors)):
            point_dict = {
                "id": ids[i],
                "vector": vectors[i],
                "payload": payloads[i],
            }
            if texts and i < len(texts):
                point_dict["vector"] = {
                    "dense": vectors[i],
                    "sparse": texts[i],
                }
            points.append(PointStruct(**point_dict))

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
        from qdrant_client.models import SparseVector, FusionQuery

        sparse_vector = self._compute_bm25(query_text)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector={
                "dense": query_vector,
                "sparse": sparse_vector,
            },
            query_filter=query_filter,
            limit=top_k,
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

    def _compute_bm25(self, text: str) -> Dict[str, Any]:
        import re
        from collections import Counter

        tokens = re.findall(r"\w+", text.lower())
        term_freq = Counter(tokens)

        max_tf = max(term_freq.values()) if term_freq else 1

        sparse_indices = []
        sparse_values = []

        for term, tf in term_freq.items():
            sparse_indices.append(hash(term) % 100000)
            sparse_values.append(tf / max_tf)

        return {
            "indices": sparse_indices,
            "values": sparse_values,
        }

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)
