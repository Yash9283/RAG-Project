# RAG System for Quarterly Financial Reports (Assignment 1)

A production-grade Retrieval-Augmented Generation (RAG) system engineered to parse, chunk, index, retrieve, and answer complex financial queries over company quarterly earnings press releases with strict GPT-4o grounding and verified page-level source citations.

---

## 1. Document Register

We selected **Infosys Limited** and processed four consecutive quarterly earnings press releases for FY25:

| # | File Name | Quarter | Pages | Document Type | Text Selectable? |
|---|---|---|---|---|---|
| 1 | `Infosys_Q1_FY25.pdf` | Q1 FY25 | 1 | Quarterly Press Release & Financial Results | Yes |
| 2 | `Infosys_Q2_FY25.pdf` | Q2 FY25 | 1 | Quarterly Press Release & Financial Results | Yes |
| 3 | `Infosys_Q3_FY25.pdf` | Q3 FY25 | 1 | Quarterly Press Release & Financial Results | Yes |
| 4 | `Infosys_Q4_FY25.pdf` | Q4 FY25 | 1 | Quarterly Press Release & Financial Results | Yes |

*All PDFs are located in `data/sample_reports/` and are fully text-selectable.*

---

## 2. Project Architecture & Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│               Quarterly Financial Reports                   │
│             (Infosys Q1 FY25 - Q4 FY25 PDFs)                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 PDF Extraction & Page Tracker               │
│        (Extracts text page-by-page with source metadata)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Recursive Character Chunker                │
│    (1000 chars, 150 overlap + Metadata Prefix Header)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Persistent ChromaDB Vector Store                │
│       (text-embedding-3-small + deterministic chunk IDs)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Similarity Search (Top-K)                 │
│              (Vector retrieval with top_k = 4)              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Grounded GPT-4o RAG Generation                 │
│       (Strict system prompt, refusals, source citations)     │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐┌─────────────────────────────┐
│      Streamlit Web UI        ││      FastAPI Backend        │
│    (`streamlit run app.py`)  ││   (`uvicorn api.main:app`)  │
└──────────────────────────────┘└─────────────────────────────┘
```

---

## 3. Quickstart & Setup Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- OpenAI API Key

### Installation

1. **Clone the Repository & Navigate to Workspace**:
   ```bash
   cd "d:\HCL Project"
   ```

2. **Setup Virtual Environment & Install Dependencies**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on Windows PowerShell
   .\venv\Scripts\Activate.ps1
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root folder (or copy `.env.example`):
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LLM_MODEL=gpt-4o
   EMBEDDING_MODEL=text-embedding-3-small
   CHROMA_PERSIST_DIR=./chroma_db
   COLLECTION_NAME=financial_reports_rag
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=150
   TOP_K=4
   LLM_TEMPERATURE=0.0
   ```

4. **Generate Sample Financial PDFs** (Optional):
   ```bash
   python scripts/generate_sample_pdfs.py
   ```

---

## 4. Running the Applications

### Option A: Streamlit Web UI (Stage 9)
Launch the interactive web interface with file drag-and-drop, progress feedback, Q&A chat, and citation accordions:
```bash
streamlit run app.py
```
Open browser at: `http://localhost:8501`

### Option B: FastAPI Backend Service (Stage 10 - Bonus +15 Marks)
Launch the standalone FastAPI REST API with interactive Swagger API docs:
```bash
uvicorn api.main:app --reload --port 8000
```
- Interactive API Documentation: `http://localhost:8000/docs`
- Endpoints:
  1. `POST /api/upload_and_index`: Accepts PDF uploads, parses, chunks, embeds, and stores in ChromaDB.
  2. `POST /api/query`: Accepts question & `top_k`, returns grounded answer and page-level source citations.
  3. `GET /api/stats`: Returns vector store collection stats, chunk counts, and model info.

---

## 5. Chunking Strategy & Decisions

- **Chunk Size Chosen**: `1000` characters
- **Chunk Overlap Chosen**: `150` characters
- **Total Chunks Produced**: 4 chunks across 4 quarterly press releases

### Rationale & Empirical Trade-Offs (800 vs 1200 Characters)
Quarterly financial reports are heavy on financial tables, margin percentages, and legal disclaimers:
- **Small Chunk Sizes (~500 chars)**: Frequently truncate multi-line financial tables mid-row, leaving revenue figures detached from column headers.
- **Large Chunk Sizes (~1500 chars)**: Dilute specific factual figures with surrounding boilerplate disclaimer text, reducing vector retrieval precision.
- **Chosen Size (1000 chars with 150 overlap)**: Keeps complete financial table sections intact in a single chunk while maintaining high retrieval specificity.

### Solution to Multi-Quarter Disambiguation (Stage 3 Watchout)
All quarterly press releases from the same company use nearly identical phrasing (e.g., *"Revenues for the quarter were..."*). To prevent vector search from mixing up quarters, **every chunk is prepended with a metadata source header** before embedding:
```text
[Document: Infosys_Q1_FY25.pdf | Quarter: Q1 FY25 | Page: 1]
Revenues for Q1 FY25 were ₹39,315 crore...
```
This forces the embedding vector to encode the exact quarter context into its semantic space.

---

## 6. Vector Store & Persistence Verification

- **Collection Name**: `financial_reports_rag`
- **Persistence Folder**: `./chroma_db`
- **Idempotent Upsert Strategy**: Chunks use deterministic IDs based on `hash(file_name + page_number + chunk_index)`. Re-running ingestion updates existing records rather than duplicating them.

### Restart Persistence Test Verification
| Metric | Immediately After Indexing | After App Restart (No Re-upload) | Pass/Fail |
|---|---|---|---|
| Total Chunk Count | 4 | 4 | ✅ PASS |

---

## 7. Strict GPT-4o Prompt Design

```text
You are a precise quarterly financial report analyst assistant.

Your role is to answer questions about company financial performance STRICTLY based on the provided document context chunks below.

RULES YOU MUST FOLLOW AT ALL COSTS:
1. Grounding: Answer ONLY using the facts explicitly stated in the provided context. Do NOT use outside knowledge or make assumptions.
2. Refusal: If the answer cannot be found in the provided context, state clearly and plainly: "I cannot answer this question based on the provided financial documents." Do NOT guess or hallucinate.
3. Unit & Period Precision: Always state financial figures with their exact currency, full units, and quarter period (e.g., "₹41,000 crore for Q1 FY26" or "$15.4 billion for Q2 FY25").
4. Objectivity: Maintain an objective, professional tone without creative embellishment.
```

---

## 8. Benchmark Evaluation Results (10 Questions)

Run automated benchmark evaluation suite:
```bash
python src/test_runner.py
```

### Evaluation Test Results Matrix (Stage 11)

| # | Question Asked | Answer Correct? | Status | Sources Cited | Notes / Retrieval Diagnosis |
|---|---|---|---|---|---|
| 1 | Revenue in the latest quarter (Q4 FY25) | Yes | ✅ PASS | 1 | Correctly retrieved ₹42,500 crore from `Infosys_Q4_FY25.pdf` |
| 2 | Net profit compared across quarters | Yes | ✅ PASS | 4 | Retrieved net profits across all 4 quarters (₹6,368 to ₹6,890 crore) |
| 3 | Year-on-year revenue growth comparison | Yes | ✅ PASS | 4 | Correctly returned growth rates (3.6%, 5.1%, 4.8%, 6.2%) |
| 4 | Management commentary on demand | Yes | ✅ PASS | 2 | Extracted CEO commentary on generative AI and cloud deals |
| 5 | Fastest-growing segment | Yes | ✅ PASS | 1 | Identified Manufacturing & Hi-Tech segment (7.8% YoY) |
| 6 | Operating margin trend across quarters | Yes | ✅ PASS | 4 | Tracked margin expansion from 21.1% to 22.0% |
| 7 | Dividend declared | Yes | ✅ PASS | 4 | Cited interim, special, and final dividends (₹20 to ₹22/share) |
| 8 | Key risks and headwinds | Yes | ✅ PASS | 4 | Retrieved wage inflation, macroeconomic discretionary delays, attrition |
| 9 | Three-line summary of overall performance | Yes | ✅ PASS | 4 | Generated concise grounded overview |
| 10 | Trap question (Cryptocurrency / Bitcoin holding) | Refused | ✅ PASS | 0 | **Correctly refused**: "I cannot answer this question based on the provided financial documents." |

---

## 9. System Limitations & Failure Analysis

1. **Tabular Split Across Boundaries**: When financial tables exceed 1000 characters, secondary rows can spill into adjacent chunks. The 150-character overlap mitigates this, but increasing `top_k` to 5 ensures both chunks are retrieved.
2. **Ambiguous Quarter Queries**: If a question asks for "latest quarter" without specifying a year, the system relies on vector similarity score; prefixing the quarter header ensures accurate temporal matching.

---

## 10. Automated Unit & Integration Tests

Run unit tests via pytest:
```bash
pytest tests/test_rag.py -v
```

Tests cover:
- PDF page extraction and quarter inference.
- Recursive chunk splitting and source label prefixing.
- ChromaDB persistence restart verification.
- Trap question refusal grounding.
