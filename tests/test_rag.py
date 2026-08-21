import os
import shutil
import pytest
from pathlib import Path
from src.pdf_processor import PDFProcessor, DocumentPage
from src.chunker import RecursiveChunker, Chunk
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.rag_chain import RAGChain

@pytest.fixture
def sample_page():
    return DocumentPage(
        file_name="Infosys_Q1_FY25.pdf",
        file_path="/dummy/path/Infosys_Q1_FY25.pdf",
        page_number=1,
        total_pages=5,
        quarter="Q1 FY25",
        text="Infosys announced revenues of ₹39,315 crore for Q1 FY25 with operating margin of 21.1%."
    )

def test_quarter_extraction():
    processor = PDFProcessor()
    assert processor.extract_quarter_from_filename("Infosys_Q1_FY25.pdf") == "Q1 FY25"
    assert processor.extract_quarter_from_filename("TCS_Q3_FY26.pdf") == "Q3 FY26"
    assert processor.extract_quarter_from_filename("random.pdf") == "Unknown Quarter"

def test_chunker_prefixing(sample_page):
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.create_chunks([sample_page])
    
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert "[Document: Infosys_Q1_FY25.pdf | Quarter: Q1 FY25 | Page: 1]" in first_chunk.text
    assert first_chunk.file_name == "Infosys_Q1_FY25.pdf"
    assert first_chunk.quarter == "Q1 FY25"
    assert first_chunk.page_number == 1

def test_vector_store_persistence(tmp_path, sample_page):
    persist_dir = str(tmp_path / "chroma_test")
    store = VectorStoreManager(persist_dir=persist_dir, collection_name="test_col")
    
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.create_chunks([sample_page])
    
    added_count = store.add_chunks(chunks)
    assert added_count == len(chunks)
    
    stats1 = store.get_stats()
    assert stats1["total_chunks"] == len(chunks)
    
    # Simulate restart test: reload collection from disk without re-indexing
    store_reloaded = VectorStoreManager(persist_dir=persist_dir, collection_name="test_col")
    stats2 = store_reloaded.get_stats()
    
    # Verify count after restart matches
    assert stats2["total_chunks"] == stats1["total_chunks"]

def test_rag_chain_trap_question(tmp_path, sample_page):
    persist_dir = str(tmp_path / "chroma_test_rag")
    store = VectorStoreManager(persist_dir=persist_dir, collection_name="test_rag_col")
    chunker = RecursiveChunker()
    chunks = chunker.create_chunks([sample_page])
    store.add_chunks(chunks)
    
    retriever = Retriever(vector_store=store)
    chain = RAGChain(retriever=retriever)
    
    # Ask trap question
    res = chain.answer("What was the CEO's personal Bitcoin holdings performance?")
    assert "cannot answer" in res["answer"].lower()
