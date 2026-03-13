import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from routers import ingest, query


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
VECTOR_SIZE = 384


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(query.router, prefix="/query", tags=["Query"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
