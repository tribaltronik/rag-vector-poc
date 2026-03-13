# rag-vector-poc

**Ask Your Docs** — A document Q&A application that lets you upload documents (PDF, TXT, Markdown) and ask questions about them using AI.


![Architecture](docs/diagram.png)

## Quick Start

### Prerequisites

You need two things installed on your computer:
1. **Docker Desktop** — Download from https://www.docker.com/products/docker-desktop
2. **Ollama model** — After starting the app, you'll download the AI model

### Starting the Application

Open your terminal (Command Prompt on Windows, Terminal on Mac) and run:

```bash
make start
```

Wait about 30 seconds for everything to load.

### Downloading the AI Model (First Time Only)

The first time you run this, you need to download the AI model. Run this command:

```bash
docker exec -it ollama ollama pull llama3.2
```

This downloads about 2GB — it may take a few minutes.

### Using the Application

1. Open your browser and go to: **http://localhost:8501**
2. You'll see the "Ask Your Docs" interface
3. Use the sidebar to upload a document (PDF, TXT, or Markdown)
4. Type a question about your document in the chat box

## Stopping the Application

To stop all services:

```bash
make stop
```

## Starting Again

```bash
make start
```

## Completely Removing Everything

If you want to delete all data and start fresh:

```bash
make destroy
```

## Troubleshooting

### "Docker not running" error
Make sure Docker Desktop is open and running (look for the whale icon in your menu bar/taskbar).

### Services not responding
Try stopping and starting again:
```bash
make stop
make start
```

### First time slow?
The AI model needs to download on first run. Check with:
```bash
docker exec -it ollama ollama list
```

## Available Services

| Service | URL | Description |
|---------|-----|-------------|
| **UI** | http://localhost:8501 | The web interface you use |
| **API** | http://localhost:8000 | The backend (for developers) |
| **Qdrant** | http://localhost:6333/dashboard | Vector database UI |

## For Developers

See [AGENTS.md](./AGENTS.md) for developer guidelines.
