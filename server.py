from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_simple import SimpleRAG, read_pdf
import os
import shutil
from google.api_core import exceptions as google_exceptions

app = FastAPI()

# ✅ Correct CORS (no credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SimpleRAG()

UPLOAD_DIR = "documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AskRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    print("UPLOAD STARTED:", file.filename)

    safe_name = os.path.basename(file.filename.replace("\\", "/"))
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_name.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    path = os.path.join(UPLOAD_DIR, safe_name)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if safe_name.lower().endswith(".pdf"):
        text = read_pdf(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    try:
        rag.ingest_text(text, source=safe_name)
    except google_exceptions.ResourceExhausted:
        raise HTTPException(
            status_code=429,
            detail="Gemini embedding quota exceeded. Add billing or wait and retry.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    return {"status": "indexed", "file": safe_name}


@app.post("/ask")
def ask(req: AskRequest):
    return {"answer": rag.ask(req.question)}


@app.get("/")
def health():
    return {"status": "ok","message": "RAG server running"}


@app.get("/sources")
def list_sources():
    return {"sources": rag.list_sources()}
 
