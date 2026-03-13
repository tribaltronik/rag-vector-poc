from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


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


class QueryFilter(BaseModel):
    document_name: Optional[str] = None
    document_id: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filter: Optional[QueryFilter] = None
    use_hybrid: bool = False
    use_rerank: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResult]


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str
