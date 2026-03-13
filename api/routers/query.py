import os
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    QuantizationConfig,
    ScalarQuantization,
)

from models.schemas import QueryRequest, QueryResponse, SearchResult
from services.embedder import Embedder
from services.vector_store import VectorStore
from services.llm import LLM
from services.reranker import Reranker


router = APIRouter()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")


def get_embedder():
    return Embedder()


def get_vector_store(request: Request):
    return VectorStore(request.app.state.qdrant_client, COLLECTION_NAME)


def get_llm():
    return LLM()


def get_reranker():
    return Reranker()


@router.post("/", response_model=QueryResponse)
async def query_documents(
    request_data: QueryRequest,
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
    llm: LLM = Depends(get_llm),
    reranker: Reranker = Depends(get_reranker),
):
    if not request_data.question or len(request_data.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    query_embedding = embedder.embed_query(request_data.question)

    query_filter = None
    if request_data.filter:
        query_filter = vector_store._build_filter(
            document_name=request_data.filter.document_name,
            document_id=request_data.filter.document_id,
        )

    if request_data.use_hybrid:
        search_results = vector_store.hybrid_search(
            query_text=request_data.question,
            query_vector=query_embedding,
            top_k=request_data.top_k * 2,
            query_filter=query_filter,
        )
    else:
        search_results = vector_store.search(
            query_vector=query_embedding,
            top_k=request_data.top_k * 2,
            query_filter=query_filter,
        )

    if not search_results:
        return QueryResponse(
            answer="No relevant documents found. Please upload some documents first.",
            sources=[],
        )

    if request_data.use_rerank and len(search_results) > 1:
        search_results = reranker.rerank(
            request_data.question,
            search_results,
            top_k=request_data.top_k,
        )

    context = "\n\n".join(
        [
            f"[{r['document_name']} - Chunk {r['chunk_index']}]: {r['text']}"
            for r in search_results[: request_data.top_k]
        ]
    )

    answer = llm.generate(request_data.question, context)

    sources = [
        SearchResult(
            chunk_id=r["id"],
            text=r["text"],
            document_id=r["document_id"],
            score=r["score"],
            document_name=r.get("document_name", ""),
        )
        for r in search_results[: request_data.top_k]
    ]

    return QueryResponse(answer=answer, sources=sources)


@router.get("/search")
async def search_documents(
    q: str,
    top_k: int = 5,
    document_name: Optional[str] = None,
    document_id: Optional[str] = None,
    use_hybrid: bool = False,
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
):
    if not q or len(q.strip()) == 0:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    query_embedding = embedder.embed_query(q)

    query_filter = vector_store._build_filter(
        document_name=document_name,
        document_id=document_id,
    )

    if use_hybrid:
        results = vector_store.hybrid_search(
            query_text=q,
            query_vector=query_embedding,
            top_k=top_k,
            query_filter=query_filter,
        )
    else:
        results = vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            query_filter=query_filter,
        )

    return {"query": q, "results": results}
