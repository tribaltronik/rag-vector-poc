import os
from typing import List
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed([query])[0]
