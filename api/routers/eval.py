import os

from fastapi import APIRouter, Depends, Request

from models.schemas import EvalRequest, EvalResponse, EvalResult, EvalMetrics
from services.embedder import Embedder
from services.vector_store import VectorStore
from services.evaluator import RetrievalEvaluator


router = APIRouter()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

_evaluator = RetrievalEvaluator()


def get_embedder():
    return Embedder()


def get_vector_store(request: Request):
    return VectorStore(request.app.state.qdrant_client, COLLECTION_NAME)


@router.post("/", response_model=EvalResponse)
async def run_evaluation(
    eval_request: EvalRequest,
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
):
    _evaluator.reset()

    results = []

    for test_case in eval_request.test_cases:
        query_embedding = embedder.embed_query(test_case.query)

        search_results = vector_store.search(
            query_vector=query_embedding,
            top_k=eval_request.top_k,
        )

        eval_result = _evaluator.evaluate_query(
            query=test_case.query,
            expected_doc_id=test_case.expected_document_id,
            retrieved_results=search_results,
            k=eval_request.top_k,
        )

        results.append(
            EvalResult(
                query=eval_result.query,
                expected_doc_id=eval_result.expected_doc_id,
                retrieved_doc_id=eval_result.retrieved_doc_id,
                relevant=eval_result.relevant,
                precision_at_k=eval_result.precision_at_k,
                recall=eval_result.recall,
            )
        )

    metrics = _evaluator.get_metrics()

    return EvalResponse(
        metrics=EvalMetrics(**metrics),
        results=results,
    )


@router.get("/metrics")
async def get_metrics():
    return _evaluator.get_metrics()


@router.post("/reset")
async def reset_evaluation():
    _evaluator.reset()
    return {"status": "reset"}
