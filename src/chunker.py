import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass
from src.pdf_processor import DocumentPage

@dataclass
class Chunk:
    chunk_id: str
    text: str          # Text with source prefix (used for embedding & search context)
    raw_text: str      # Original chunk text without prefix
    source_label: str  # Descriptive header label
    file_name: str
    page_number: int
    quarter: str
    chunk_index: int

class RecursiveChunker:
    """
    Recursively splits text into chunks of specified size and overlap.
    Prefixes each chunk with file name and quarter context to differentiate
    similar sentences across multiple quarterly financial reports.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        """
        Recursively splits raw text using separators.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunks.append(text[start:].strip())
                break
                
            # Try to break at logical separators
            sub_text = text[start:end]
            split_pos = -1
            for sep in self.separators:
                if sep == "":
                    split_pos = len(sub_text)
                    break
                pos = sub_text.rfind(sep)
                if pos != -1 and pos > (self.chunk_size // 3):
                    split_pos = pos + len(sep)
                    break
                    
            if split_pos == -1 or split_pos == 0:
                split_pos = len(sub_text)
                
            chunk_str = sub_text[:split_pos].strip()
            if chunk_str:
                chunks.append(chunk_str)
                
            # Advance start index considering overlap
            start += max(1, split_pos - self.chunk_overlap)
            
        return chunks

    def create_chunks(self, pages: List[DocumentPage]) -> List[Chunk]:
        """
        Processes document pages into prefixed, metadata-rich Chunk objects.
        Uses stable deterministic IDs based on file_name, page_number, and chunk_index.
        """
        all_chunks: List[Chunk] = []
        
        for page in pages:
            if not page.text.strip():
                continue
                
            raw_chunks = self._split_text(page.text)
            for idx, raw_chunk in enumerate(raw_chunks):
                source_label = f"[Document: {page.file_name} | Quarter: {page.quarter} | Page: {page.page_number}]"
                prefixed_text = f"{source_label}\n{raw_chunk}"
                
                # Deterministic stable chunk ID for idempotency & persistence restarts
                id_seed = f"{page.file_name}_p{page.page_number}_c{idx}_{hashlib.md5(raw_chunk.encode('utf-8')).hexdigest()[:8]}"
                chunk_id = id_seed.replace(' ', '_').replace('.', '_')
                
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=prefixed_text,
                        raw_text=raw_chunk,
                        source_label=source_label,
                        file_name=page.file_name,
                        page_number=page.page_number,
                        quarter=page.quarter,
                        chunk_index=idx
                    )
                )
                
        return all_chunks
