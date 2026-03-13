import os

from fastapi import APIRouter, HTTPException, Depends, Request

from models.schemas import QueryRequest, QueryResponse, SearchResult
from services.embedder import Embedder
from services.vector_store import VectorStore
from services.llm import LLM


router = APIRouter()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")


def get_embedder():
    return Embedder()


def get_vector_store(request: Request):
    return VectorStore(request.app.state.qdrant_client, COLLECTION_NAME)


def get_llm():
    return LLM()


@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
    llm: LLM = Depends(get_llm),
):
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    query_embedding = embedder.embed_query(request.question)

    search_results = vector_store.search(
        query_vector=query_embedding, top_k=request.top_k
    )

    if not search_results:
        return QueryResponse(
            answer="No relevant documents found. Please upload some documents first.",
            sources=[],
        )

    context = "\n\n".join(
        [
            f"[{r['document_name']} - Chunk {r['chunk_index']}]: {r['text']}"
            for r in search_results
        ]
    )

    answer = llm.generate(request.question, context)

    sources = [
        SearchResult(
            chunk_id=r["id"],
            text=r["text"],
            document_id=r["document_id"],
            score=r["score"],
            document_name=r.get("document_name", ""),
        )
        for r in search_results
    ]

    return QueryResponse(answer=answer, sources=sources)


@router.get("/search")
async def search_documents(
    q: str,
    top_k: int = 5,
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
):
    if not q or len(q.strip()) == 0:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    query_embedding = embedder.embed_query(q)

    results = vector_store.search(query_vector=query_embedding, top_k=top_k)

    return {"query": q, "results": results}
