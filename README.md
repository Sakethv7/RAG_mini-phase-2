# RAG-Mini — Simple Document Q&A (Gemini + Vector Store)

A from-scratch Retrieval-Augmented Generation (RAG) system for querying documents. Uses Google Gemini for embeddings and answer generation by default, with an optional local embedder and either a built-in NumPy store or Qdrant.

No LangChain.
No heavyweight orchestration.
Embeddings are API-based by default; set `GOOGLE_API_KEY` before running.



🧩 Architecture Overview
React UI (Vite)
   │
   │  REST API (Axios)
   ▼
FastAPI Backend
   │
   ├─ Gemini embeddings (API) or local `sentence-transformers`
   ├─ Vector store: NumPy (local) or Qdrant (cloud)
   ├─ Retrieval (cosine similarity)
   ├─ Gemini (answer synthesis)
   └─ Optional S3 backup for uploaded documents

## Features
- **Document formats:** `.txt`, `.md`, `.pdf`
- **Smart chunking:** 800 chars with 200 overlap (configurable)
- **Semantic search:** Gemini embeddings (default) or local `sentence-transformers` + cosine similarity
- **Vector store:** tiny NumPy index on disk (`./lite_index`) or Qdrant Cloud (`VECTOR_BACKEND=qdrant`)
- **Document backup:** optional S3 upload of original files on ingest
- **Gemini answers:** concise responses grounded in retrieved context
- **CLI chat:** ask questions in your terminal

✨ What This Project Does

Upload documents (.pdf, .txt, .md)

Chunk + embed them with Gemini embeddings

Store vectors in a lightweight NumPy index

Retrieve the most relevant chunks per query via cosine similarity

Generate structured, grounded answers using Gemini

Show sources used for every answer

Render clean Markdown summaries in a React UI

This is a real, production-style RAG architecture, just stripped down to essentials.

📄 Document Ingestion

Supports PDF, Markdown, and Text

OCR-normalized (fixes broken PDF spacing)

Smart chunking (800 chars, 200 overlap)

🔍 Retrieval

Semantic search using Gemini embeddings (or local models) with cosine similarity via the selected vector backend.

🤖 Answer Generation

Gemini 2.5 Flash by default, grounded in retrieved context with concise formatting.

🖥️ UI

React + Vite

Markdown-rendered answers

Collapsible Sources panel

Clean chat experience

---

## Quick Start

### 1) Clone & setup
```bash
git clone https://github.com/Sakethv7/RAG_mini
cd RAG-mini
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here
```

### Embedding options
- Default: Gemini embeddings (`EMBED_PROVIDER=gemini`, `GEMINI_EMBED_MODEL=models/embedding-001`), requires `GOOGLE_API_KEY`.
- Local (no embedding API calls): set `EMBED_PROVIDER=local` and optionally `LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2`. Install the extra deps: `pip install sentence-transformers`.

### Vector store options
- Local NumPy (default): set `VECTOR_BACKEND=local`. Data lives in `./lite_index`.
- Qdrant: set `VECTOR_BACKEND=qdrant` and configure:
  - `QDRANT_URL` (e.g., `https://<cluster>.cloud.qdrant.io:6333`)
  - `QDRANT_API_KEY`
  - `QDRANT_COLLECTION` (e.g., `documents`)

### Document storage (optional)
- To back up uploaded files to S3, set:
  - `S3_BUCKET`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
  - Optional `S3_PREFIX` (folder prefix)
- `UPLOAD_DIR` and `INDEX_DIR` can be overridden via env if you mount a disk.
