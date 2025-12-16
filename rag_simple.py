# rag_simple.py — Local NumPy RAG + Gemini (Render-safe, latest-doc aware)

import os
import re
import json
import hashlib
from typing import List
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai


# =========================
# Helpers
# =========================

def normalize_ocr_text(text: str) -> str:
    text = re.sub(r"(?<=\w)\s(?=\w)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size=800, overlap=200):
    out, i = [], 0
    while i < len(text):
        j = min(i + chunk_size, len(text))
        out.append(text[i:j])
        i = j - overlap if j < len(text) else j
    return out


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def point_id(source: str, chunk_id: int) -> int:
    h = hashlib.blake2b(f"{source}:{chunk_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=True)


# =========================
# Lite Vector Store
# =========================

class LiteVectorStore:
    def __init__(self, index_dir="./lite_index"):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        self.vectors = None
        self.ids = None
        self.payloads = []

        self._load()

    def _load(self):
        try:
            self.vectors = np.load(f"{self.index_dir}/vectors.npy")
            self.ids = np.load(f"{self.index_dir}/ids.npy")
            with open(f"{self.index_dir}/payloads.json") as f:
                self.payloads = json.load(f)
        except:
            self.vectors, self.ids, self.payloads = None, None, []

    def _save(self):
        if self.vectors is None:
            return
        np.save(f"{self.index_dir}/vectors.npy", self.vectors)
        np.save(f"{self.index_dir}/ids.npy", self.ids)
        with open(f"{self.index_dir}/payloads.json", "w") as f:
            json.dump(self.payloads, f)

    def upsert(self, ids, vectors, payloads):
        if self.vectors is None:
            self.vectors = vectors
            self.ids = ids
            self.payloads = payloads
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.ids = np.concatenate([self.ids, ids])
            self.payloads.extend(payloads)
        self._save()

    def search(self, qvec, k=5, source_filter=None):
        sims = np.dot(self.vectors, qvec)
        idxs = np.argsort(-sims)

        results = []
        for i in idxs:
            p = self.payloads[i]
            if source_filter and p["source"] != source_filter:
                continue
            results.append({**p, "score": float(sims[i])})
            if len(results) == k:
                break

        return results

    def reset(self):
        self.vectors, self.ids, self.payloads = None, None, []
        self._save()


# =========================
# SimpleRAG (Gemini-only embeddings)
# =========================

class SimpleRAG:
    def __init__(self, index_dir="./lite_index"):
        load_dotenv()

        self.store = LiteVectorStore(index_dir)
        self.meta_path = f"{index_dir}/meta.json"

        self.latest_source = None
        self._load_meta()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)

        self.chat_model = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash",
            system_instruction=(
                "You are a document analysis assistant.\n"
                "Use ONLY the provided context.\n"
                "If information is missing, say so clearly.\n\n"
                "For summaries:\n"
                "- Start with a short overview paragraph\n"
                "- Then use markdown headings and bullet points"
            )
        )

    # -------- Metadata --------

    def _load_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                self.latest_source = json.load(f).get("latest_source")

    def _save_meta(self):
        with open(self.meta_path, "w") as f:
            json.dump({"latest_source": self.latest_source}, f)

    # -------- Embeddings (Gemini) --------

    def embed(self, texts: List[str]) -> np.ndarray:
        print("Embedding", len(texts), "chunks") 

        vectors = []
        for t in texts:
            r = genai.embed_content(
                model="models/embedding-001",
                content=t
            )
            vectors.append(r["embedding"])
        return np.array(vectors, dtype=np.float32)

    # -------- Ingestion --------

    def ingest_text(self, text: str, source: str):
        text = normalize_ocr_text(text)
        chunks = chunk_text(text)

        vectors = self.embed(chunks)
        now = datetime.utcnow().isoformat()

        self.latest_source = source
        self._save_meta()

        ids = np.array([point_id(source, i) for i in range(len(chunks))])
        payloads = [
            {
                "source": source,
                "chunk_id": i,
                "text": ch,
                "timestamp": now
            }
            for i, ch in enumerate(chunks)
        ]

        self.store.upsert(ids, vectors, payloads)

    # -------- Retrieval --------

    def retrieve(self, question, k=5):
        qv = self.embed([question])[0]

        if self.latest_source:
            hits = self.store.search(qv, k=k, source_filter=self.latest_source)
            if hits:
                return hits

        return self.store.search(qv, k=k)

    # -------- Answer --------

    def ask(self, question, return_chunks=False):
        contexts = self.retrieve(question)

        if not contexts:
            answer = "No relevant information found in the indexed documents."
            return (answer, []) if return_chunks else answer

        context_text = "\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in contexts
        )

        prompt = f"""
Use the context below to answer clearly and accurately.

Question:
{question}

Context:
{context_text}
"""

        resp = self.chat_model.generate_content(prompt)
        answer = resp.text.strip()

        return (answer, contexts) if return_chunks else answer

    def recreate_collection(self):
        self.store.reset()
        self.latest_source = None
        self._save_meta()
