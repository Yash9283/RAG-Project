import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.config import Config
from src.chunker import Chunk

logger = logging.getLogger(__name__)

class FastLocalEmbeddingFunction(chromadb.EmbeddingFunction):
    def name(self) -> str:
        return "fast_local_embedding"

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = []
        for text in input:
            vec = [0.0] * 384
            words = text.lower().split()
            for w in words:
                idx = abs(hash(w)) % 384
                vec[idx] += 1.0
            norm = (sum(x * x for x in vec) ** 0.5) or 1.0
            vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings

class VectorStoreManager:
    """
    Manages persistent ChromaDB vector store operations:
    - Initializing persistent disk collection
    - Idempotent indexing of chunks with stable IDs
    - Persistence restart testing
    - Vector similarity retrieval
    """
    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or Config.COLLECTION_NAME
        
        # Ensure persistence directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Select embedding function
        if Config.OPENAI_API_KEY and not Config.OPENAI_API_KEY.startswith("mock-"):
            try:
                import chromadb.utils.embedding_functions as ef
                self.embedding_fn = ef.OpenAIEmbeddingFunction(
                    api_key=Config.OPENAI_API_KEY,
                    model_name=Config.EMBEDDING_MODEL
                )
            except Exception:
                self.embedding_fn = FastLocalEmbeddingFunction()
        else:
            self.embedding_fn = FastLocalEmbeddingFunction()
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Quarterly Financial Reports RAG Store"}
        )

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Adds chunks to ChromaDB using stable chunk IDs.
        If chunk IDs already exist, upsert overwrites them cleanly to avoid duplicates.
        """
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "file_name": c.file_name,
                "page_number": c.page_number,
                "quarter": c.quarter,
                "chunk_index": c.chunk_index,
                "source_label": c.source_label,
                "raw_text": c.raw_text
            }
            for c in chunks
        ]

        # Upsert ensures idempotent insertion without duplicates
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant chunk documents for a query.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, max(1, self.collection.count()))
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                    "file_name": meta.get("file_name", "Unknown"),
                    "page_number": meta.get("page_number", 0),
                    "quarter": meta.get("quarter", "Unknown"),
                    "snippet": meta.get("raw_text", doc)
                })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns collection stats for monitoring & Stage 10 API.
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_dir": str(self.persist_dir),
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.LLM_MODEL
        }

    def clear_collection(self) -> None:
        """
        Clears all items in the collection.
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Quarterly Financial Reports RAG Store"}
        )
