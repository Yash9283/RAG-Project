from typing import List, Dict, Any, Optional
from src.vector_store import VectorStoreManager
from src.config import Config

class Retriever:
    """
    Handles similarity search retrieval over indexed financial documents in ChromaDB.
    """
    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant chunk snippets for the query.
        """
        k = top_k if top_k is not None else Config.TOP_K
        return self.vector_store.search(query, top_k=k)

    def print_retrieved_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Helper method to format and print retrieved chunks for manual inspection & debugging.
        """
        print(f"\n--- Retrieved {len(chunks)} Chunks ---")
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            file_name = meta.get("file_name", "Unknown")
            page = meta.get("page_number", "Unknown")
            quarter = meta.get("quarter", "Unknown")
            dist = chunk.get("distance", 0.0)
            snippet = chunk.get("snippet", chunk.get("document", ""))
            print(f"[{idx}] {file_name} (Page {page}, {quarter}) | Distance: {dist:.4f}")
            print(f"    Snippet: {snippet[:200]}...\n")
