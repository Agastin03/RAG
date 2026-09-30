# Advanced Enterprise RAG Chatbot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-orange.svg)](https://www.trychroma.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestrated-green.svg)](https://langchain-ai.github.io/langgraph/)
[![BM25](https://img.shields.io/badge/Hybrid-BM25%20%2B%20Dense-purple.svg)](https://github.com/dorianbrown/rank_bm25)

An advanced, enterprise-grade **Retrieval-Augmented Generation (RAG)** AI Chatbot built in Python following state-of-the-art AI Agent workflows.

The chatbot answers questions **strictly based** on the official [Agentic AI eBook Knowledge Base](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf). It uses multi-query expansion, hybrid dense/sparse retrieval, Reciprocal Rank Fusion (RRF), cross-score reranking, and LangGraph faithfulness validation to guarantee zero hallucination.

---

## 1. System Architecture & Workflows

### Dynamic PDF Switch & Ingestion Flow
```mermaid
flowchart LR
    classDef pdf1 fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff
    classDef pdf2 fill:#1e293b,stroke:#10b981,stroke-width:2px,color:#fff
    classDef purge fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#fff
    classDef chat fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#fff

    subgraph Step1 ["📥 STEP 1: INITIAL PDF INGESTION"]
        A1["📄 Upload document1.pdf<br/>(POST /upload)"]:::pdf1 --> A2["⚙️ PyMuPDF extract_pages()<br/>Extract Page Text & Metadata"]:::pdf1
        A2 --> A3["✂️ split_text_recursively()<br/>Chunk Size 1000, Overlap 150"]:::pdf1
        A3 --> A4["🧠 EmbeddingService<br/>384d Vectors (all-MiniLM-L6-v2)"]:::pdf1
        A4 --> A5["💾 ChromaStore.add_chunks()<br/>Indexed in ChromaDB"]:::pdf1
    end

    subgraph Step2 ["🔄 STEP 2: DYNAMIC PDF SWITCH & PURGE"]
        B1["📄 Upload document2.pdf<br/>(POST /upload)"]:::pdf2 --> B2["🔥 ChromaStore.reset_collection()<br/>delete_collection() & create_collection()"]:::purge
        B2 --> B3["🧹 ChromaDB Collection:<br/>Purged & Wiped EMPTY"]:::purge
        B3 --> B4["⚙️ PyMuPDF extract_pages()<br/>Extract PDF #2 Page Text"]:::pdf2
        B4 --> B5["✂️ split_text_recursively()<br/>Chunk Size 1000, Overlap 150"]:::pdf2
        B5 --> B6["🧠 EmbeddingService<br/>Generate 384d Vectors for PDF #2"]:::pdf2
        B6 --> B7["💾 ChromaStore.add_chunks()<br/>Store PDF #2 Chunks in ChromaDB"]:::pdf2
    end

    subgraph Step3 ["🕸️ STEP 3: ADVANCED LANGGRAPH RAG WORKFLOW"]
        C1["🔍 User Question<br/>(POST /chat)"]:::chat --> N1["1️⃣ Node: understand_query<br/>Multi-Query Variations"]:::chat
        N1 --> N2["2️⃣ Node: hybrid_retrieve<br/>ChromaDB Dense + BM25 Sparse + RRF"]:::chat
        N2 --> N3["3️⃣ Node: rerank_context<br/>Score Top-5 Chunks from PDF #2"]:::chat
        N3 --> N4["4️⃣ Node: evaluate_relevance<br/>Confidence Threshold >= 0.50"]:::chat
        N4 --> COND{"Is Relevant?"}:::chat
        COND -->|Yes| N5["5️⃣ Node: generate_answer<br/>Gemini System Grounding Prompt"]:::chat
        COND -->|No| N6["6️⃣ Node: format_refusal<br/>Return Grounded Refusal Message"]:::chat
        N5 --> N7["7️⃣ Node: validate_faithfulness<br/>Verify Zero Hallucination"]:::chat
        N7 --> OUT["✨ Final Response<br/>(Answers ONLY from PDF #2)"]:::chat
        N6 --> OUT
    end

    Step1 --> Step2
    Step2 --> Step3
```

---

## 2. Key Enterprise Features

- **Multi-Query Expansion**: Rephrases and expands input queries into multiple domain-specific variations to maximize retrieval recall.
- **Hybrid Dense + Sparse Retrieval**: Combines semantic embeddings (`all-MiniLM-L6-v2` via ChromaDB) with sparse lexical keyword matching (`BM25Okapi` via `rank_bm25`).
- **Reciprocal Rank Fusion (RRF)**: Merges dense vector and sparse keyword rankings using $RRF(d) = \sum \frac{1}{60 + rank(d)}$.
- **Context Reranking**: Re-scores candidate context chunks using multi-token overlap and positional relevance.
- **Strict Grounding & Refusal**: Refuses to answer queries outside the knowledge base using clear refusal messaging without hallucinating.
- **LangGraph Faithfulness Validation**: Validates generated responses to ensure full factual fidelity to retrieved context.

---

## 3. Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Agentic-AI.pdf        # Target eBook PDF document
│
├── app/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Environment settings & thresholds
│   ├── ingestion.py          # PyMuPDF PDF extraction & chunking
│   ├── embeddings.py         # Dense vector embedding service
│   ├── vectorstore.py        # ChromaDB persistent store management
│   ├── retrieval.py          # Multi-Query, BM25, RRF Fusion & Reranking
│   ├── graph.py              # LangGraph workflow state machine
│   ├── llm.py                # Gemini LLM with strict grounding prompt
│   ├── ui.html               # Interactive Web UI interface
│   └── main.py               # FastAPI REST API & endpoints
│
├── scripts/
│   └── ingest.py             # Ingestion script
│
├── tests/
│   └── test_rag.py           # Automated test suite (13 test cases)
│
├── chroma_db/                # Persistent ChromaDB storage
├── .env.example              # Environment variable template
├── .env                      # Local environment variables
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10+
- Git

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Agastin03/RAG.git
   cd RAG
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env`:
   ```bash
   # Windows
   copy .env.example .env

   # Linux / macOS
   cp .env.example .env
   ```

   Set your Gemini API key in `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-3.1-flash-lite
   CHROMA_PERSIST_DIRECTORY=./chroma_db
   CHROMA_COLLECTION_NAME=agentic_ai_knowledge
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   TOP_K=5
   RELEVANCE_THRESHOLD=0.50
   ```

---

## 5. PDF Ingestion

Run the ingestion script to process `Agentic-AI.pdf` into ChromaDB:

```bash
python scripts/ingest.py
```

---

## 6. Running the API & Web UI

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive Web UI**: `http://127.0.0.1:8000`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs`

---

## 7. API Endpoint Documentation

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "service": "Agentic AI RAG Chatbot API",
    "version": "1.0.0",
    "total_indexed_chunks": 123
  }
  ```

### 2. PDF Upload
- **Endpoint**: `POST /upload`
- Upload any PDF file dynamically to re-index ChromaDB.

### 3. Chat Query
- **Endpoint**: `POST /chat`
- **Request**: `{"question": "What is Agentic AI?"}`
- **Response**: Includes grounded answer, retrieved context chunks with page numbers, and confidence score.

---

## 8. Automated Testing

Run the pytest suite:
```bash
pytest tests/test_rag.py -v
```

All 13 test cases covering Web UI, PDF upload, health check, multi-query expansion, RRF fusion, context reranking, graph routing, and sample questions pass successfully.
