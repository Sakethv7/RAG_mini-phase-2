# RAG-Mini — Simple Document Q&A (Gemini + Local Vector Store)

A minimal, from-scratch Retrieval-Augmented Generation (RAG) system.  
No LangChain. No Qdrant required. Works fully offline for embeddings and uses **Google Gemini** for answers.

## Features
- **Document formats:** `.txt`, `.md`, `.pdf`
- **Smart chunking:** 800 chars with 200 overlap (configurable)
- **Semantic search:** `sentence-transformers/all-MiniLM-L6-v2` (local, no API)
- **Local vector store:** tiny NumPy index on disk (`./lite_index`) — no services to run
- **Gemini answers:** concise responses grounded in retrieved context
- **CLI chat:** ask questions in your terminal
- **Robustness:** prompt budgets + retries + extractive fallback so you always get something

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
