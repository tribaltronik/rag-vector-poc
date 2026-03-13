import os
import json
import httpx
from typing import List, Dict, Any


class LLM:
    def __init__(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.model = os.getenv("LLM_MODEL", "llama3.2")

    def generate(self, prompt: str, context: str = "") -> str:
        full_prompt = self._build_prompt(prompt, context)

        payload = {"model": self.model, "prompt": full_prompt, "stream": False}

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{self.ollama_host}/api/generate", json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a helpful AI assistant. Use the following context from documents to answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{self.ollama_host}/api/chat", json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
