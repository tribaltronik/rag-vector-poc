import os
import json
import httpx
from typing import List, Dict, Any, Optional


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

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)
