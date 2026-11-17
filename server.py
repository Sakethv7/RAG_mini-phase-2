from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_simple import SimpleRAG, read_pdf
from decisions import GuardUnavailable
import os
import shutil

app = FastAPI()

# ✅ Correct CORS (no credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-ui-frontend.onrender.com"],
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

    if not file.filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if file.filename.lower().endswith(".pdf"):
        text = read_pdf(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    rag.ingest_text(text, source=file.filename)

    return {"status": "indexed", "file": file.filename}


@app.post("/ask")
def ask(req: AskRequest):
    return {"answer": rag.ask(req.question)}


@app.post("/ask/guarded")
def ask_guarded(req: AskRequest):
    try:
        return rag.ask_guarded(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except GuardUnavailable:
        raise HTTPException(status_code=503, detail="Guardrail unavailable")
    except Exception as e:
        print("ask_guarded upstream failure:", e)
        raise HTTPException(status_code=502, detail="Upstream model call failed")


@app.post("/ask/verified")
def ask_verified(req: AskRequest):
    """
    Chain-of-Verification variant: slower (3-4 Gemini calls) but flags
    when the answer makes claims the retrieved context doesn't support.
    Use for high-stakes queries where hallucination cost is high.
    """
    return rag.ask_verified(req.question)


@app.get("/")
def health():
    return {"status": "ok","message": "RAG server running"}
 