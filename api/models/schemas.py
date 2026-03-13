from pydantic import BaseModel
from typing import Optional, List


class DocumentUpload(BaseModel):
    filename: str
    content_type: str


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    document_id: str
    chunk_index: int


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    document_id: str
    score: float
    document_name: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResult]


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str
