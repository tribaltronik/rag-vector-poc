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


class EvalTestCase(BaseModel):
    query: str
    expected_document_id: str


class EvalRequest(BaseModel):
    test_cases: List[EvalTestCase]
    top_k: int = 5


class EvalResult(BaseModel):
    query: str
    expected_doc_id: str
    retrieved_doc_id: str
    relevant: bool
    precision_at_k: float
    recall: float


class EvalMetrics(BaseModel):
    total_queries: int
    precision_at_k_avg: float
    recall_avg: float
    hit_rate: float


class EvalResponse(BaseModel):
    metrics: EvalMetrics
    results: List[EvalResult]
