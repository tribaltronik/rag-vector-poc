# 🧠 Vector Database POC

> **Goal:** Demonstrate real-world understanding of vector embeddings, semantic search, and LLM-augmented retrieval (RAG) in a fully containerized local setup.

---

## 🎯 Use Case: Semantic Document Q&A with RAG

**"Ask Your Docs"** — A Retrieval-Augmented Generation (RAG) pipeline that lets users upload documents (PDF, Markdown, plain text), index them into a vector database, and query them using natural language. An LLM synthesizes answers grounded in the retrieved document chunks.

### Why this impresses interviewers:
- Shows understanding of **embeddings, vector similarity, and semantic search** (not just keyword search)
- Demonstrates **LLM integration** patterns (RAG vs fine-tuning tradeoffs)
- Highlights **microservices thinking** via containerized services
- Covers **real production concerns**: chunking strategy, metadata filtering, re-ranking
- Fully runnable **locally** — no cloud bills, no secrets needed

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │          │    │              │    │                     │   │
│  │  Streamlit│───▶│  FastAPI     │───▶│  Qdrant             │   │
│  │   UI     │    │  Backend     │    │  (Vector DB)        │   │
│  │ :8501    │    │  :8000       │    │  :6333 / :6334      │   │
│  └──────────┘    └──────┬───────┘    └─────────────────────┘   │
│                         │                                       │
│                         │                                       │
│                  ┌──────▼───────┐    ┌─────────────────────┐   │
│                  │              │    │                     │   │
│                  │  Ollama      │    │  PostgreSQL          │   │
│                  │  (Local LLM) │    │  (Metadata Store)   │   │
│                  │  :11434      │    │  :5432              │   │
│                  └──────────────┘    └─────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Services at a glance

| Service      | Image                     | Purpose                                  |
|-------------|---------------------------|------------------------------------------|
| `qdrant`    | `qdrant/qdrant`           | Vector storage and similarity search     |
| `ollama`    | `ollama/ollama`           | Local LLM inference (no API key needed!) |
| `api`       | Custom FastAPI image      | Ingestion pipeline + query endpoint      |
| `ui`        | Custom Streamlit image    | User-facing chat + upload interface      |
| `postgres`  | `postgres:16-alpine`      | Document metadata, tags, upload history  |

---

## 🗂️ Repository Structure

```
vector-db-poc/
├── README.md
├── docker-compose.yml
├── docker-compose.kind.yml          # Kubernetes Kind alternative
├── .env.example
│
├── api/                             # FastAPI backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── routers/
│   │   ├── ingest.py               # Document upload & chunking
│   │   └── query.py                # Search + RAG endpoint
│   ├── services/
│   │   ├── embedder.py             # Embedding model wrapper
│   │   ├── vector_store.py         # Qdrant client abstraction
│   │   ├── chunker.py              # Text splitting strategies
│   │   └── llm.py                  # Ollama chat wrapper
│   └── models/
│       └── schemas.py              # Pydantic models
│
├── ui/                              # Streamlit frontend
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
│
├── k8s/                             # Kind / Kubernetes manifests
│   ├── namespace.yaml
│   ├── qdrant/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── ollama/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── api/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   ├── ui/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── postgres/
│       ├── deployment.yaml
│       ├── service.yaml
│       └── pvc.yaml
│
├── scripts/
│   ├── seed_data.py                 # Load sample documents automatically
│   └── eval_retrieval.py           # Simple retrieval quality evaluation
│
└── docs/
    ├── architecture.md
    ├── chunking-strategies.md
    └── interview-talking-points.md
```

---

## 🐳 Option A: Docker Compose (Recommended for simplicity)

### `docker-compose.yml`

```yaml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  postgres:
    image: postgres:16-alpine
    container_name: postgres
    environment:
      POSTGRES_USER: poc
      POSTGRES_PASSWORD: poc
      POSTGRES_DB: vector_poc
    volumes:
      - pg_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    # For GPU support, add:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]

  api:
    build: ./api
    container_name: api
    ports:
      - "8000:8000"
    environment:
      QDRANT_HOST: qdrant
      QDRANT_PORT: 6333
      OLLAMA_HOST: http://ollama:11434
      POSTGRES_URL: postgresql://poc:poc@postgres:5432/vector_poc
      EMBED_MODEL: all-MiniLM-L6-v2    # HuggingFace sentence-transformer
      LLM_MODEL: llama3.2              # or mistral, phi3, gemma2
      COLLECTION_NAME: documents
    depends_on:
      - qdrant
      - ollama
      - postgres

  ui:
    build: ./ui
    container_name: ui
    ports:
      - "8501:8501"
    environment:
      API_BASE_URL: http://api:8000
    depends_on:
      - api

volumes:
  qdrant_data:
  pg_data:
  ollama_models:
```

### Quick Start (Docker Compose)

```bash
# 1. Clone and configure
git clone https://github.com/your-username/vector-db-poc
cd vector-db-poc
cp .env.example .env

# 2. Start all services
docker compose up -d

# 3. Pull the LLM model (one-time, ~2GB)
docker exec -it ollama ollama pull llama3.2

# 4. Seed with sample documents
python scripts/seed_data.py

# 5. Open the UI
open http://localhost:8501
```

---

## ☸️ Option B: Kubernetes with Kind (Advanced — great interview talking point)

### Why Kind?
- Runs a real multi-node Kubernetes cluster locally via Docker
- Forces you to reason about **Deployments, Services, PVCs, ConfigMaps**
- Demonstrates readiness for **production Kubernetes** environments
- Easy to show multi-replica Qdrant with StatefulSets

### Setup

```bash
# Install kind
brew install kind  # or use the binary from kind.sigs.k8s.io

# Create cluster
kind create cluster --name vector-poc --config k8s/kind-config.yaml

# Load locally built images
kind load docker-image api:latest --name vector-poc
kind load docker-image ui:latest  --name vector-poc

# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/qdrant/
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/ollama/
kubectl apply -f k8s/api/
kubectl apply -f k8s/ui/

# Port-forward to access locally
kubectl port-forward svc/ui 8501:8501 -n vector-poc
kubectl port-forward svc/qdrant 6333:6333 -n vector-poc
```

### `k8s/kind-config.yaml`

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
```

---

## 🔬 Core Implementation Details

### 1. Document Ingestion Pipeline (`api/routers/ingest.py`)

```
Upload file
    │
    ▼
Extract text (PyMuPDF for PDF, plain text, markdown)
    │
    ▼
Chunk text (RecursiveCharacterTextSplitter, chunk_size=512, overlap=64)
    │
    ▼
Generate embeddings (sentence-transformers: all-MiniLM-L6-v2, 384 dims)
    │
    ▼
Store vector + payload in Qdrant
    │
    ▼
Store metadata (filename, chunk count, upload date) in PostgreSQL
```

**Interview talking point:** Explain chunking strategy tradeoffs:
- Smaller chunks → better precision, worse context
- Larger chunks → more context, noisier retrieval
- Overlap → avoids cutting concepts in half

### 2. Query & RAG Pipeline (`api/routers/query.py`)

```
User question
    │
    ▼
Embed question (same model as ingestion!)
    │
    ▼
Qdrant similarity search (top-k=5, cosine distance)
    │
    ▼
Optional: metadata filter (e.g. by document, date range)
    │
    ▼
Build prompt: [System] + [Retrieved chunks] + [Question]
    │
    ▼
Ollama LLM generates answer with citations
    │
    ▼
Return answer + source document references
```

**Interview talking point:** Why RAG over fine-tuning?
- No training cost, instant knowledge updates
- Traceable sources (reduces hallucinations)
- Works with proprietary/private documents

### 3. Qdrant Collection Setup

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="qdrant", port=6333)

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=384,           # all-MiniLM-L6-v2 output dims
        distance=Distance.COSINE
    )
)
```

**Interview talking point:** Distance metrics — Cosine vs. Dot Product vs. Euclidean, and when to use each.

---

## 📊 Features to Implement (Prioritized)

### Phase 1 — Core MVP ✅
- [x] Upload PDF/TXT/MD documents via UI
- [x] Automatic text extraction and chunking
- [x] Embedding generation and Qdrant indexing
- [x] Semantic search endpoint (top-k results)
- [x] RAG Q&A with Ollama LLM
- [x] Source citation in answers

### Phase 2 — Depth (Good interview conversations)
- [x] Metadata filtering (filter by filename, date)
- [x] Hybrid search: combine BM25 keyword + vector (Qdrant supports this natively!)
- [x] Multiple embedding model support (swap via env var)
- [x] Re-ranking step with a cross-encoder model
- [x] Qdrant Web UI integration (`http://localhost:6333/dashboard`)
- [x] Integration tests

### Phase 3 — Production Signals (Really impresses)
- [ ] Prometheus metrics + Grafana dashboard via Docker Compose
- [ ] Async ingestion queue (Celery + Redis)
- [ ] `/eval` endpoint: run retrieval evaluation against a test set
- [ ] Horizontal scaling demo with Kind (multiple API replicas)
- [ ] Helm chart for Kubernetes deployment

---

## 🗣️ Interview Talking Points

### "Why Qdrant over Pinecone/Weaviate/Chroma?"
> Qdrant is self-hostable, production-grade, supports both REST and gRPC, has a beautiful dashboard UI, and supports payload filtering alongside vector search. Pinecone is managed-only (no local), Chroma is great for quick prototypes but less production-hardened, Weaviate adds more complexity with its schema model.

### "What's the hardest part of building a RAG system?"
> Chunking strategy and retrieval quality. Getting the right chunk size, handling document structure (tables, headers), and evaluating whether retrieved chunks actually answer the question. I built a simple evaluation script that measures retrieval recall against a labeled test set.

### "How would you scale this in production?"
> Qdrant supports distributed mode with sharding and replication. I'd move to a StatefulSet in Kubernetes with persistent volumes, add a read replica for search traffic, and decouple ingestion via an async queue (Kafka or SQS) to handle burst uploads without blocking queries.

### "Why not just use `pgvector`?"
> `pgvector` is excellent for adding vector search to an existing Postgres-centric stack. Dedicated vector databases like Qdrant are purpose-built for ANN (Approximate Nearest Neighbor) at scale with better indexing algorithms (HNSW), payload filtering, and higher query throughput when vectors are the primary access pattern.

---

## 🛠️ Tech Stack Summary

| Layer           | Technology                            | Rationale                              |
|----------------|---------------------------------------|----------------------------------------|
| Vector DB       | Qdrant                                | Self-hosted, fast HNSW, great UI       |
| Embeddings      | `all-MiniLM-L6-v2` (HuggingFace)     | Lightweight, high quality, no API key  |
| LLM             | Ollama + Llama 3.2 / Mistral          | Fully local, no cost, no data leakage  |
| Backend         | FastAPI + Python                      | Async, typed, auto-docs via OpenAPI    |
| Frontend        | Streamlit                             | Fast to build, great for demos         |
| Metadata DB     | PostgreSQL                            | Familiar, relational metadata store    |
| Orchestration   | Docker Compose / Kind (k8s)           | Local dev + production-style option    |
| Container build | Docker + multi-stage builds           | Lean final images                      |

---

## 📦 Sample Documents to Seed (scripts/seed_data.py)

Use publicly available documents for a compelling demo:
- **Wikipedia articles** (e.g., History of Python, Machine Learning)
- **ArXiv paper abstracts** (AI/ML papers)
- **OpenAI / Anthropic public blog posts** (retrieved via requests)
- **Your own resume / portfolio** — ultra-impressive to demo querying your own CV!

---

## ⚠️ Known Limitations & Future Work

> Being upfront about tradeoffs is a sign of engineering maturity. This section is intentional — use it as a conversation starter with interviewers.

### 🧱 Technical Limitations

| # | Limitation | Impact | Suggested Fix |
|---|-----------|--------|---------------|
| 1 | **Naive fixed-size chunking** | Splits tables, code blocks, and structured content poorly | Add a `strategy` param: `fixed`, `sentence`, `markdown_header` |
| 2 | **No retrieval evaluation** | Can't quantify whether the right chunks are being retrieved | Add a `/eval` endpoint with a labeled Q&A test set and recall score |
| 3 | **Single-stage retrieval** | Cosine similarity alone loses precision on complex queries | Add re-ranking with a cross-encoder (e.g. `ms-marco-MiniLM`) or HyDE |
| 4 | **Fixed embedding model** | `all-MiniLM-L6-v2` underperforms on domain-specific content (legal, code, medical) | Make model configurable via env var; benchmark alternatives |
| 5 | **Ollama is CPU-bound without GPU** | Inference can take 5–20s on a laptop, making live demos sluggish | Pre-warm model on startup; add a loading spinner in UI; document GPU flag |

### 🏗️ Architecture Limitations

| # | Limitation | Impact | Suggested Fix |
|---|-----------|--------|---------------|
| 6 | **Synchronous ingestion** | Large PDFs block the API and can time out | Decouple with Celery + Redis or an async task queue |
| 7 | **Single Qdrant instance** | No replication — one crash = data inaccessible | Use Qdrant distributed mode with replicas (StatefulSet in k8s) |
| 8 | **No authentication or multi-tenancy** | All users share all documents; no access control | Add JWT auth + per-user Qdrant collection namespacing |
| 9 | **No observability** | Hard to debug bad answers or slow queries in production | Add structured logging (`structlog`), tracing (OpenTelemetry), and a `/health` endpoint |
| 10 | **Streamlit as the UI** | Signals "prototype" to engineering-focused interviewers | Optionally replace with a minimal React/Next.js frontend |

### 📦 Portfolio / Differentiation Limitations

| # | Limitation | Suggested Fix |
|---|-----------|---------------|
| 11 | **"Ask Your Docs" is a saturated demo** | Add a memorable twist — seed it with your own **resume/CV** and let interviewers query *you* |
| 12 | **PostgreSQL is underused** | Earn its presence: track upload history, chunk stats, query logs, and per-doc retrieval scores |

---

### 🗺️ Recommended Roadmap to Address These

```
Phase 1 (MVP)         Phase 2 (Depth)              Phase 3 (Production Signals)
──────────────        ──────────────────           ─────────────────────────────
Core RAG pipeline  →  Chunking strategies      →   Async ingestion queue
Upload + search       Retrieval evaluation         Auth + multi-tenancy
Ollama local LLM      Re-ranking step              Prometheus + Grafana
Docker Compose        Hybrid search (BM25+vec)     Horizontal scaling (Kind)
                      Swap embedding models        OpenTelemetry tracing
```

---

### 💬 How to Use This in an Interview

When asked *"What would you improve?"*, lead with:

> *"I intentionally documented the limitations in the repo. The two I'd prioritize in a production setting are: first, moving to document-aware chunking because fixed-size splitting degrades quality on structured content; and second, adding a retrieval evaluation harness — without measuring recall you're flying blind. I also designed the embedding model to be swappable via env var so you can benchmark alternatives without code changes."*

This signals **production thinking**, not just tutorial-following.

---

## 🔗 Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama Model Library](https://ollama.com/library)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Kind Quick Start](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/)

---

*Built as a curriculum portfolio project to demonstrate practical AI infrastructure knowledge.*
