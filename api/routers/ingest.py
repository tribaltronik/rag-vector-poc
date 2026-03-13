import os
import uuid
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from pypdf import PdfReader

from models.schemas import IngestResponse
from services.chunker import Chunker
from services.embedder import Embedder
from services.vector_store import VectorStore


router = APIRouter()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")


def get_embedder():
    return Embedder()


def get_vector_store(request: Request):
    return VectorStore(request.app.state.qdrant_client, COLLECTION_NAME)


def get_chunker():
    return Chunker()


@router.post("/", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
    chunker: Chunker = Depends(get_chunker),
):
    if file.content_type not in [
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    ]:
        raise HTTPException(
            status_code=400, detail="Unsupported file type. Supported: PDF, TXT, MD"
        )

    document_id = str(uuid.uuid4())
    filename = file.filename

    content = await file.read()
    text = extract_text(content, file.content_type)

    if not text or len(text.strip()) == 0:
        raise HTTPException(
            status_code=400, detail="Could not extract text from document"
        )

    chunks = chunker.chunk_text(text)

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created from document")

    embeddings = embedder.embed(chunks)

    payloads = [
        {
            "text": chunk,
            "document_id": document_id,
            "document_name": filename,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]

    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    vector_store.add_vectors(embeddings, payloads, ids)

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        chunks_created=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks",
    )


def extract_text(content: bytes, content_type: str) -> str:
    if content_type == "application/pdf":
        return extract_pdf(content)
    else:
        return content.decode("utf-8", errors="ignore")


def extract_pdf(content: bytes) -> str:
    pdf_file = BytesIO(content)
    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text
