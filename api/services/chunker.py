import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] = ["\n\n", "\n", " ", ""],
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

    def chunk_documents(self, documents: List[str]) -> List[str]:
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk_text(doc))
        return chunks
