# RAG Vector Database POC

**Ask Your Docs** — A document Q&A application that lets you upload documents (PDF, TXT, Markdown) and ask questions about them using AI.

![Architecture](docs/diagram.png)

## Features

### Phase 1 - Core MVP
- Upload PDF/TXT/MD documents via UI
- Automatic text extraction and chunking
- Embedding generation and Qdrant indexing
- Semantic search endpoint (top-k results)
- RAG Q&A with Ollama LLM
- Source citation in answers

### Phase 2 - Enhanced Search
- Metadata filtering (filter by filename, document_id)
- Hybrid search (BM25 + vector)
- Multiple embedding model support (configurable via env var)
- Re-ranking with cross-encoder

### Phase 3 - Production Ready
- Prometheus metrics + Grafana dashboard
- Retrieval evaluation endpoint (`/eval`)
- Comprehensive observability

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │          │    │              │    │                     │    │
│  │ Streamlit│───▶│  FastAPI     │───▶│ Qdrant              │    │
│  │   UI     │    │  Backend     │    │ (Vector DB)         │    │
│  │ :8501    │    │  :8000       │    │ :6333 / :6334       │    │
│  └──────────┘    └──────┬───────┘    └─────────────────────┘    │
│                         │                                       │
│                  ┌──────▼───────┐    ┌─────────────────────┐    │
│                  │              │    │                     │    │
│                  │  Ollama      │    │ PostgreSQL          │    │
│                  │ (Local LLM)  │    │ (Metadata Store)    │    │
│                  │  :11434      │    │ :5432               │    │
│                  └──────────────┘    └─────────────────────┘    │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────────┐                    │
│  │ Prometheus   │    │ Grafana             │                    │
│  │ :9090        │    │ :3000               │                    │
│  └──────────────┘    └─────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

You need two things installed on your computer:
1. **Docker Desktop** — Download from https://www.docker.com/products/docker-desktop

### Starting the Application

Open your terminal and run:

```bash
make start
```

Wait about 30 seconds for everything to load.

### Downloading the AI Model (First Time Only)

```bash
docker exec -it ollama ollama pull llama3.2
```

This downloads about 2GB — it may take a few minutes.

### Using the Application

1. Open your browser and go to: **http://localhost:8501**
2. You'll see the "Ask Your Docs" interface
3. Use the sidebar to upload a document (PDF, TXT, or Markdown)
4. Type a question about your document in the chat box

## Available Services

| Service | URL | Description |
|---------|-----|-------------|
| **UI** | http://localhost:8501 | The web interface you use |
| **API** | http://localhost:8000 | The backend API |
| **API Docs** | http://localhost:8000/docs | OpenAPI documentation |
| **Qdrant** | http://localhost:6333/dashboard | Vector database UI |
| **Prometheus** | http://localhost:9090 | Metrics backend |
| **Grafana** | http://localhost:3000 | Dashboards (admin/admin) |

## API Endpoints

### Query with RAG
```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "top_k": 5}'
```

### Search with Filters
```bash
curl "http://localhost:8000/query/search?q=certificate&document_name=file.pdf"
```

### Hybrid Search
```bash
curl "http://localhost:8000/query/search?q=test&use_hybrid=true"
```

### Retrieval Evaluation
```bash
curl -X POST http://localhost:8000/eval/ \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {"query": "certificate", "expected_document_id": "doc-id-here"}
    ],
    "top_k": 3
  }'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | qdrant | Qdrant server |
| `QDRANT_PORT` | 6333 | Qdrant port |
| `OLLAMA_HOST` | http://ollama:11434 | Ollama URL |
| `POSTGRES_URL` | postgresql://poc:poc@postgres:5432/vector_poc | PostgreSQL |
| `EMBED_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `LLM_MODEL` | llama3.2 | Ollama model |
| `COLLECTION_NAME` | documents | Qdrant collection |
| `PROMETHEUS_ENABLED` | false | Enable Prometheus metrics |

## Commands

```bash
make start    # Start all services
make stop     # Stop all services
make destroy # Stop and remove all containers and volumes
make logs    # View logs
make health  # Check service health
```

## For Developers

See [AGENTS.md](./AGENTS.md) for developer guidelines.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Vector DB | Qdrant |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (llama3.2) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Metadata DB | PostgreSQL |
| Monitoring | Prometheus + Grafana |
| Monitoring | Prometheus + Grafana |
