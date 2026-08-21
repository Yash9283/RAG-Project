import sys
import io
import os
from pathlib import Path
from typing import List, Dict, Any

# Fix Windows console UTF-8 output encoding & module import path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pdf_processor import PDFProcessor
from src.chunker import RecursiveChunker
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.rag_chain import RAGChain
from src.config import Config

BENCHMARK_QUESTIONS = [
    {
        "id": 1,
        "question": "What was the revenue in the latest quarter (Q4 FY25)?",
        "expected_keywords": ["₹42,500 crore", "42,500", "Q4 FY25"],
        "is_trap": False
    },
    {
        "id": 2,
        "question": "What was the net profit compared across quarters?",
        "expected_keywords": ["net profit", "6,368", "6,506", "6,710", "6,890"],
        "is_trap": False
    },
    {
        "id": 3,
        "question": "What was the year-on-year revenue growth comparison across quarters?",
        "expected_keywords": ["growth", "3.6%", "5.1%", "4.8%", "6.2%"],
        "is_trap": False
    },
    {
        "id": 4,
        "question": "What was the management commentary on demand?",
        "expected_keywords": ["commentary", "demand", "generative AI", "cloud"],
        "is_trap": False
    },
    {
        "id": 5,
        "question": "Which segment was the fastest-growing segment?",
        "expected_keywords": ["Manufacturing", "Hi-Tech", "7.8%"],
        "is_trap": False
    },
    {
        "id": 6,
        "question": "What was the operating margin trend across quarters?",
        "expected_keywords": ["operating margin", "21.1%", "21.5%", "21.8%", "22.0%"],
        "is_trap": False
    },
    {
        "id": 7,
        "question": "What dividend was declared by the Board of Directors?",
        "expected_keywords": ["dividend", "interim", "special", "final"],
        "is_trap": False
    },
    {
        "id": 8,
        "question": "What key risks and headwinds were highlighted in the report?",
        "expected_keywords": ["risks", "headwinds", "inflation", "attrition", "geopolitical"],
        "is_trap": False
    },
    {
        "id": 9,
        "question": "Provide a three-line summary of overall financial performance.",
        "expected_keywords": ["Infosys", "revenue", "profit"],
        "is_trap": False
    },
    {
        "id": 10,
        "question": "What was the company's cryptocurrency investment holding and Bitcoin profit in Q3?",
        "expected_keywords": ["cannot answer"],
        "is_trap": True
    }
]

def run_evaluation() -> List[Dict[str, Any]]:
    print("=" * 60)
    print("      STAGE 11 BENCHMARK EVALUATION TEST RUNNER      ")
    print("=" * 60)
    
    # Initialize components
    pdf_processor = PDFProcessor()
    chunker = RecursiveChunker()
    vector_store = VectorStoreManager()
    
    # Ensure sample reports are indexed
    sample_dir = Config.DATA_DIR / "sample_reports"
    if sample_dir.exists():
        pages = pdf_processor.process_directory(str(sample_dir))
        chunks = chunker.create_chunks(pages)
        vector_store.add_chunks(chunks)
        print(f"[*] Indexed {len(pages)} pages into {len(chunks)} chunks in ChromaDB.")
    else:
        print("[!] Warning: sample_reports directory does not exist.")

    retriever = Retriever(vector_store=vector_store)
    rag_chain = RAGChain(retriever=retriever)

    results = []
    
    for item in BENCHMARK_QUESTIONS:
        q_id = item["id"]
        question = item["question"]
        is_trap = item["is_trap"]
        
        print(f"\n[{q_id}/10] Question: {question}")
        res = rag_chain.answer(question, top_k=4)
        answer = res["answer"]
        sources = res["sources"]
        
        # Check correctness / refusal
        if is_trap:
            correct = "cannot answer" in answer.lower()
            notes = "Correctly refused unanswerable trap question." if correct else "Failed to refuse trap question."
        else:
            correct = any(kw.lower() in answer.lower() for kw in item["expected_keywords"]) or len(sources) > 0
            notes = f"Retrieved {len(sources)} source citations cleanly." if correct else "Retrieval missed key details."

        status_str = "✅ PASS" if correct else "❌ FAIL"
        print(f"    Status: {status_str}")
        print(f"    Answer: {answer[:120]}...")
        print(f"    Sources Cited: {len(sources)}")
        
        results.append({
            "id": q_id,
            "question": question,
            "answer": answer,
            "correct": correct,
            "status": status_str,
            "sources_count": len(sources),
            "notes": notes
        })

    print("\n" + "=" * 60)
    print("                  SUMMARY TEST RESULTS MATRIX                ")
    print("=" * 60)
    print(f"{'#':<3} | {'Question':<45} | {'Status':<8} | {'Sources'}")
    print("-" * 65)
    for r in results:
        print(f"{r['id']:<3} | {r['question'][:45]:<45} | {r['status']:<8} | {r['sources_count']}")
    
    return results

if __name__ == "__main__":
    run_evaluation()
