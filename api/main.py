import os
from contextlib import asynccontextmanager
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from routers import ingest, query, eval


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
VECTOR_SIZE = 384

http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

ingest_documents_total = Counter("ingest_documents_total", "Total documents ingested")

queries_total = Counter("queries_total", "Total queries processed")

embeddings_generated_total = Counter(
    "embeddings_generated_total", "Total embeddings generated"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    app.state.qdrant_client = client

    yield

    pass


app = FastAPI(
    title="Document Q&A API",
    description="RAG-powered document Q&A with Qdrant and Ollama",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    import time

    method = request.method
    path = request.url.path

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    http_requests_total.labels(
        method=method, endpoint=path, status=response.status_code
    ).inc()

    http_request_duration_seconds.labels(method=method, endpoint=path).observe(duration)

    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prometheus_enabled = os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"
if prometheus_enabled:
    prometheus_app = make_asgi_app()
    app.mount("/metrics", prometheus_app)

app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(eval.router, prefix="/eval", tags=["Evaluation"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
