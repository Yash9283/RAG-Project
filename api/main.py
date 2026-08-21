import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import Config
from src.pdf_processor import PDFProcessor
from src.chunker import RecursiveChunker
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.rag_chain import RAGChain

app = FastAPI(
    title="Financial Reports RAG API",
    description="FastAPI Backend for Quarterly Financial Report Querying with Strict Grounding & Source Citations",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Service Instances
pdf_processor = PDFProcessor()
chunker = RecursiveChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
vector_store = VectorStoreManager()
retriever = Retriever(vector_store=vector_store)
rag_chain = RAGChain(retriever=retriever)

UPLOAD_DIR = Config.DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class QueryRequest(BaseModel):
    question: str = Field(..., example="What was the revenue in the latest quarter?")
    top_k: Optional[int] = Field(default=4, ge=1, le=10, example=4)

class SourceCitation(BaseModel):
    file_name: str
    page_number: int
    quarter: str
    snippet: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    retrieved_chunks_count: int

class IndexResponse(BaseModel):
    status: str
    files_processed: int
    chunks_created: int
    collection_name: str
    total_chunks_in_store: int

class StatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    embedding_model: str
    llm_model: str
    persist_dir: str

@app.get("/")
def read_root():
    return {
        "message": "Financial Reports RAG API Service",
        "docs_url": "/docs",
        "status": "running"
    }

@app.post("/api/upload_and_index", response_model=IndexResponse)
async def upload_and_index(files: List[UploadFile] = File(...)):
    """
    Stage 10 Endpoint 1: Uploads PDF files, processes pages, splits into prefixed chunks,
    and indexes them into persistent ChromaDB.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    saved_paths = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF.")
            
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(str(file_path))

    # Process extracted pages
    all_pages = []
    for path in saved_paths:
        pages = pdf_processor.process_pdf(path)
        all_pages.extend(pages)

    # Chunking & source prefixing
    chunks = chunker.create_chunks(all_pages)

    # Index into ChromaDB
    chunks_added = vector_store.add_chunks(chunks)
    stats = vector_store.get_stats()

    return IndexResponse(
        status="success",
        files_processed=len(saved_paths),
        chunks_created=chunks_added,
        collection_name=Config.COLLECTION_NAME,
        total_chunks_in_store=stats["total_chunks"]
    )

@app.post("/api/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Stage 10 Endpoint 2: Accepts question and top_k, retrieves relevant document chunks,
    and returns GPT-4o grounded answer with exact source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag_chain.answer(question=request.question, top_k=request.top_k)
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        retrieved_chunks_count=result["retrieved_chunks_count"]
    )

@app.get("/api/stats", response_model=StatsResponse)
def get_stats():
    """
    Stage 10 Endpoint 3: Returns vector store & model runtime statistics.
    """
    stats = vector_store.get_stats()
    return StatsResponse(
        collection_name=stats["collection_name"],
        total_chunks=stats["total_chunks"],
        embedding_model=stats["embedding_model"],
        llm_model=stats["llm_model"],
        persist_dir=stats["persist_dir"]
    )

@app.post("/api/reset")
def reset_store():
    """
    Resets the ChromaDB vector store collection.
    """
    vector_store.clear_collection()
    return {"status": "success", "message": "Collection cleared."}
