# AGENTS.md - Developer Guidelines for RAG Vector PoC

This document provides guidelines for AI agents working on this codebase.

## Project Overview

A RAG-powered document Q&A application using:
- **API**: FastAPI with Qdrant (vector store), Ollama (LLM), PostgreSQL
- **UI**: Streamlit
- **Infrastructure**: Docker Compose

## Build, Lint, and Test Commands

### Python Environment Setup

```bash
# Install API dependencies
cd api && pip install -r requirements.txt

# Install UI dependencies
cd ui && pip install -r requirements.txt
```

### Linting (Ruff)

```bash
# Run ruff linter on entire project
ruff check .

# Run ruff with auto-fix
ruff check --fix .

# Format with ruff
ruff format .
```

### Running the Application

```bash
# Start all services with Docker Compose
docker-compose up -d

# Start only specific services
docker-compose up -d qdrant postgres ollama  # Dependencies
docker-compose up -d api                     # API on port 8000
docker-compose up -d ui                       # UI on port 8501
```

### Running Tests

```bash
# Install test dependencies
pip install pytest requests qdrant-client langchain langchain-community sentence-transformers==2.3.1

# Run all tests
pytest tests/ -v

# Run unit tests only
PYTHONPATH=./api pytest tests/test_phase2.py -v

# Run integration tests (requires running services)
pytest tests/test_api_integration.py -v

# Run a single test file
pytest tests/test_phase2.py::TestMetadataFiltering -v

# Run a single test function
pytest tests/test_phase2.py::TestMetadataFiltering::test_build_filter_with_document_name -v
```

### API Development Server

```bash
# Run API locally (requires Qdrant, Postgres, Ollama running)
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

Copy `.env.example` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server | `qdrant` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `OLLAMA_HOST` | Ollama URL | `http://ollama:11434` |
| `POSTGRES_URL` | PostgreSQL connection | `postgresql://poc:poc@postgres:5432/vector_poc` |
| `EMBED_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `LLM_MODEL` | Ollama model | `llama3.2` |
| `COLLECTION_NAME` | Qdrant collection | `documents` |
| `API_BASE_URL` | API URL (UI) | `http://api:8000` |

---

## Code Style Guidelines

### General Principles

- Write clean, readable, and maintainable Python code
- Follow PEP 8 with 88-character line limit (ruff default)
- Use type hints where beneficial
- Keep functions focused and single-purpose

### Imports

```python
# Standard library first, then third-party, then local
import os
import uuid
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pypdf import PdfReader

from models.schemas import IngestResponse
from services.chunker import Chunker
```

- Use absolute imports from package root
- Group imports by type with blank lines between groups
- Sort imports alphabetically within groups

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Variables | snake_case | `document_id`, `query_embedding` |
| Functions | snake_case | `get_embedder()`, `extract_text()` |
| Classes | PascalCase | `Embedder`, `VectorStore`, `LLM` |
| Constants | UPPER_SNAKE_CASE | `COLLECTION_NAME`, `VECTOR_SIZE` |
| Modules | snake_case | `embedder.py`, `vector_store.py` |

### Type Hints

```python
from typing import List, Optional

def embed(self, texts: List[str]) -> List[List[float]]:
    ...

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class SearchResult(BaseModel):
    chunk_id: str
    text: str
    document_id: str
    score: float
    document_name: Optional[str] = None
```

### Pydantic Models

Use Pydantic v2 for request/response schemas:

```python
from pydantic import BaseModel

class DocumentUpload(BaseModel):
    filename: str
    content_type: str
```

### FastAPI Patterns

**Routers**: Define routers in `api/routers/` with dependency injection:

```python
router = APIRouter()

def get_embedder():
    return Embedder()

@router.post("/", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    embedder: Embedder = Depends(get_embedder),
):
    ...
```

**Error Handling**: Use HTTPException for user-facing errors:

```python
from fastapi import HTTPException

if not request.question or len(request.question.strip()) == 0:
    raise HTTPException(status_code=400, detail="Question cannot be empty")
```

**Lifespan**: Use async context manager for startup/shutdown:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    client = QdrantClient(...)
    app.state.qdrant_client = client
    yield
    # Shutdown (if needed)
```

### Database/Service Patterns

- Initialize heavy clients (Qdrant, models) once, store in `app.state`
- Use environment variables for configuration with sensible defaults
- Group related functionality in `services/` directory

### Code Organization

```
api/
├── main.py              # FastAPI app, lifespan, CORS
├── models/
│   └── schemas.py       # Pydantic models
├── routers/
│   ├── ingest.py        # Document upload endpoints
│   └── query.py         # Query/search endpoints
└── services/
    ├── chunker.py       # Text chunking
    ├── embedder.py      # Sentence embeddings
    ├── llm.py           # LLM generation
    └── vector_store.py  # Qdrant operations
```

### Logging

- Use standard `print` for simple cases
- For production, consider Python's `logging` module

### Testing Patterns

When adding tests:

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
```

- Place tests in `tests/` directory
- Use `pytest` with fixtures for common setup
- Mock external services (Qdrant, Ollama) where appropriate

---

## Common Development Tasks

### Adding a New Endpoint

1. Create or update schema in `api/models/schemas.py`
2. Add route handler in appropriate router file
3. Add dependency injection for services
4. Return appropriate Pydantic response model

### Adding a New Service

1. Create file in `api/services/`
2. Implement business logic in a class
3. Use environment variables for configuration
4. Add type hints for all public methods

### Modifying Data Models

1. Update Pydantic schema in `api/models/schemas.py`
2. Ensure backward compatibility when possible
3. Update corresponding database/vector store logic

---

## Docker Development Notes

- All services communicate via Docker network names (not localhost)
- API: `http://api:8000`
- Qdrant: `http://qdrant:6333`
- Ollama: `http://ollama:11434`
- PostgreSQL: `postgresql://postgres:5432`
