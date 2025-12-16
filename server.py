from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_simple import SimpleRAG, read_pdf
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-ui-frontend.onrender.com",
    ],
    allow_credentials=False,   # 🔥 IMPORTANT
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
    if not file.filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if file.filename.endswith(".pdf"):
        text = read_pdf(path)
    else:
        text = open(path, "r", encoding="utf-8").read()

    rag.ingest_text(text, source=file.filename)

    return {"status": "indexed", "file": file.filename}


@app.post("/ask")
def ask(req: AskRequest):
    answer = rag.ask(req.question)
    return {"answer": answer}


@app.get("/")
def health():
    return {"status": "ok", "message": "RAG server running"}