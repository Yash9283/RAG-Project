import os
import streamlit as st
from pathlib import Path
from src.config import Config
from src.pdf_processor import PDFProcessor
from src.chunker import RecursiveChunker
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.rag_chain import RAGChain

# Page Configuration
st.set_page_config(
    page_title="Financial Report RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 0.375rem;
        margin-bottom: 0.8rem;
    }
    .stat-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.4rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize System Components in Session State
@st.cache_resource
def get_rag_components():
    pdf_processor = PDFProcessor()
    chunker = RecursiveChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    vector_store = VectorStoreManager()
    retriever = Retriever(vector_store=vector_store)
    rag_chain = RAGChain(retriever=retriever)
    return pdf_processor, chunker, vector_store, retriever, rag_chain

pdf_processor, chunker, vector_store, retriever, rag_chain = get_rag_components()

# Session State for History
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
    st.title("Settings & Controls")
    
    st.subheader("1. Index Documents")
    uploaded_files = st.file_uploader(
        "Upload Quarterly Financial PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload press releases or fact sheets (e.g. Infosys_Q1_FY25.pdf)"
    )
    
    top_k = st.slider("Retrieval Top-K Chunks", min_value=1, max_value=8, value=Config.TOP_K)
    
    if st.button("🚀 Index Uploaded PDFs", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("⚠️ Please select at least one PDF file before indexing.")
        else:
            with st.spinner("Processing PDFs, extracting text & building vector store..."):
                upload_dir = Config.DATA_DIR / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                saved_paths = []
                for file in uploaded_files:
                    path = upload_dir / file.name
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_paths.append(str(path))
                
                # Extract & Chunk
                all_pages = []
                for p in saved_paths:
                    all_pages.extend(pdf_processor.process_pdf(p))
                
                chunks = chunker.create_chunks(all_pages)
                chunks_count = vector_store.add_chunks(chunks)
                
                st.success(f"✅ Successfully indexed {len(saved_paths)} PDF(s) into {chunks_count} prefixed chunks!")

    st.markdown("---")
    st.subheader("📊 Collection Statistics")
    stats = vector_store.get_stats()
    st.metric("Total Indexed Chunks", stats["total_chunks"])
    st.write(f"**Embedding Model:** `{stats['embedding_model']}`")
    st.write(f"**LLM Model:** `{stats['llm_model']}`")
    
    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        vector_store.clear_collection()
        st.session_state.history = []
        st.success("Database cleared!")
        st.rerun()

# Main UI Area
st.markdown('<div class="main-header">📈 Quarterly Financial Reports RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Query earnings press releases with strict GPT-4o grounding and verified page-level source citations.</div>', unsafe_allow_html=True)

# Question Input Section
stats = vector_store.get_stats()
is_db_empty = stats["total_chunks"] == 0

if is_db_empty:
    st.info("💡 **Getting Started:** Upload and index your quarterly financial report PDFs using the sidebar on the left, or use the sample reports in `data/sample_reports/`.")

# Q&A Submission Form
with st.form(key="qa_form", clear_on_submit=False):
    question_input = st.text_input(
        "Enter your financial question:",
        placeholder="e.g. What was the net profit and revenue in Q2 FY25 compared across quarters?",
        disabled=is_db_empty
    )
    submit_button = st.form_submit_button("🔍 Query System", use_container_width=False, type="primary")

if submit_button:
    if is_db_empty:
        st.warning("⚠️ Cannot ask questions before indexing documents. Please upload PDFs first.")
    elif not question_input.strip():
        st.warning("⚠️ Please type a non-empty question.")
    else:
        with st.spinner("Retrieving top chunks & generating grounded answer..."):
            result = rag_chain.answer(question=question_input.strip(), top_k=top_k)
            st.session_state.history.insert(0, result)

# Display Current & Previous Q&A Results
if st.session_state.history:
    st.markdown("---")
    st.subheader("💡 Q&A Results & Source Citations")
    
    for item in st.session_state.history:
        with st.container():
            st.markdown(f"### ❓ **Question:** {item['question']}")
            
            # Highlight Refusals gracefully
            if "cannot answer" in item['answer'].lower():
                st.warning(f"🚫 **Answer (Refusal):** {item['answer']}")
            else:
                st.success(f"📌 **Grounded Answer:**\n\n{item['answer']}")
            
            # Render Source Citations
            if item.get("sources"):
                with st.expander(f"📚 View {len(item['sources'])} Source Citation(s)", expanded=True):
                    for idx, src in enumerate(item["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source #{idx}:</strong> <code>{src['file_name']}</code> | 
                            <strong>Quarter:</strong> {src['quarter']} | 
                            <strong>Page:</strong> {src['page_number']}
                            <br/><br/>
                            <em>Excerpt snippet:</em> "{src['snippet']}"
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.caption("No sources cited for this response.")
            st.markdown("---")
