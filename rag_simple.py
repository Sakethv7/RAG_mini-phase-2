# rag_simple.py — Local NumPy RAG + Gemini (Render-safe, ENV-driven)

import os
import re
import json
import hashlib
from typing import List
from dataclasses import asdict
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai

from decisions import backends, guard, rerank, route


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
        except Exception:
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
# SimpleRAG (Gemini)
# =========================

class SimpleRAG:
    def __init__(self, index_dir="./lite_index", check_index=True):
        load_dotenv()

        # --- ENV ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        self.chat_model_name = os.getenv(
            "GEMINI_MODEL", "models/gemini-3.5-flash-lite"
        )
        # Embedding model: Gemini gemini-embedding-001, truncated to EMBED_DIM
        # and L2-normalized so dot product == cosine similarity (index: ./lite_index).
        self.embed_model_name = os.getenv(
            "GEMINI_EMBED_MODEL", "models/gemini-embedding-001"
        )
        self.embed_dim = int(os.getenv("GEMINI_EMBED_DIM", "768"))

        genai.configure(api_key=api_key)

        self.store = LiteVectorStore(index_dir)
        self.meta_path = f"{index_dir}/meta.json"

        self.latest_source = None
        self.meta_embed_model = None
        backends()  # fail at startup on a bad *_BACKEND value
        self._load_meta()
        if check_index:
            self._check_index_matches_embedder()

        self.chat_model = genai.GenerativeModel(
            model_name=self.chat_model_name,
            system_instruction=(
                "You are a document analysis assistant.\n"
                "Use ONLY the provided context.\n"
                "If information is missing, say so clearly."
            )
        )

    # -------- Metadata --------

    def _load_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                meta = json.load(f)
            self.latest_source = meta.get("latest_source")
            self.meta_embed_model = meta.get("embed_model")

    def _save_meta(self):
        with open(self.meta_path, "w") as f:
            json.dump(
                {
                    "latest_source": self.latest_source,
                    "embed_model": self.embed_model_name,
                    "embed_dim": self.embed_dim,
                },
                f,
            )

    def _check_index_matches_embedder(self):
        v = self.store.vectors
        if v is None:
            return
        if v.shape[1] != self.embed_dim or self.meta_embed_model != self.embed_model_name:
            raise RuntimeError(
                f"Index in {self.store.index_dir} has {v.shape[1]}-dim vectors "
                f"(model: {self.meta_embed_model or 'unknown'}), but the embedder is "
                f"{self.embed_model_name} at {self.embed_dim} dims. "
                "Run SimpleRAG(check_index=False).reindex() after setting the model, or restore a matching index."
            )

    def reindex(self):
        """Re-embed every stored chunk with the current embedder. Chunk text is kept."""
        texts = [p["text"] for p in self.store.payloads]
        self.store.vectors = self.embed(texts)
        self.store._save()
        self._save_meta()

    # -------- Embeddings --------

    def embed(self, texts: List[str]) -> np.ndarray:
        print("Embedding", len(texts), "chunks")

        vectors = []
        # One request per 100 texts (API batch limit) to stay under the
        # free-tier 100 requests/minute quota.
        for i in range(0, len(texts), 100):
            try:
                r = genai.embed_content(
                    model=self.embed_model_name,
                    content=texts[i:i + 100],
                    output_dimensionality=self.embed_dim,
                )
                vectors.extend(r["embedding"])
            except Exception as e:
                print(f"❌ Gemini embedding failed at chunk {i}: {e}")
                raise

        arr = np.array(vectors, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

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

    def ask(self, question):
        contexts = self.retrieve(question)

        if not contexts:
            return "No relevant information found in the indexed documents."

        context_text = "\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in contexts
        )

        prompt = f"""
Question:
{question}

Context:
{context_text}
"""

        resp = self.chat_model.generate_content(prompt)
        return resp.text.strip()

    def ask_guarded(self, question: str) -> dict:
        """
        Guard -> route -> retrieve -> answer (ADR-2, ADR-3). Raises ValueError on a bad
        question and decisions.GuardUnavailable if the guard cannot decide (fail closed).
        Each step runs on its own backend (llm, local, off), see decisions.py.
        """
        question = question.strip()
        if not question or len(question) > 2000:
            raise ValueError("Question is empty or too long")

        # No system instruction: it would bias the classification and chat prompts.
        plain = genai.GenerativeModel(model_name=self.chat_model_name)
        llm = lambda p: plain.generate_content(p).text

        trace = []
        g = guard(question, llm)
        trace.append(asdict(g))
        if g.label != "SAFE":
            return {"answer": "I can't help with that question.", "refused": g.label,
                    "found": False, "sources": [], "trace": trace}

        r = route(question, llm)
        trace.append(asdict(r))
        if r.label == "CHAT":
            answer = llm(
                "You are a friendly assistant for a document Q&A app. "
                f"Reply in one or two sentences.\n\nUser: {question}"
            ).strip()
            return {"answer": answer, "refused": None, "found": True, "sources": [], "trace": trace}

        top_n = 12 if r.label == "SUMMARY" else 5
        if backends()["rerank"] == "off":
            contexts = self.retrieve(question, k=top_n)
        else:
            contexts, rr = rerank(question, self.retrieve(question, k=20), top_n, llm)
            trace.append(asdict(rr))
        floor = float(os.getenv("SCORE_FLOOR", "0.0"))
        if not contexts or contexts[0]["score"] < floor:
            return {"answer": "No relevant information found in the indexed documents.",
                    "refused": None, "found": False, "sources": [], "trace": trace}

        context_text = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in contexts)
        answer = self.chat_model.generate_content(
            f"\nQuestion:\n{question}\n\nContext:\n{context_text}\n"
        ).text.strip()
        sources = [{"source": c["source"], "chunk_id": c["chunk_id"], "score": round(c["score"], 3)}
                   for c in contexts]
        return {"answer": answer, "refused": None, "found": True, "sources": sources, "trace": trace}

    def ask_verified(self, question: str) -> dict:
        """
        Chain-of-Verification (CoV): generate an initial answer, derive
        verification questions from it, answer each from the retrieved context
        independently, then flag any contradictions.

        Returns:
            {
                "answer": str,           # initial answer (or refined if contradictions found)
                "verified": bool,        # True if no contradictions detected
                "contradictions": list,  # [{question, conflicting_answer}, ...]
                "verification_pairs": list,  # [{question, answer}, ...]
            }

        Why: a single LLM pass can hallucinate confidently. CoV catches cases
        where the answer makes claims the retrieved context doesn't actually
        support — without fine-tuning or a separate judge model.
        """
        contexts = self.retrieve(question)

        if not contexts:
            return {
                "answer": "No relevant information found in the indexed documents.",
                "verified": False,
                "contradictions": [],
                "verification_pairs": [],
            }

        context_text = "\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in contexts
        )

        # Step 1: initial answer
        initial_answer = self.chat_model.generate_content(
            f"Question:\n{question}\n\nContext:\n{context_text}"
        ).text.strip()

        # Step 2: derive verification questions from the answer
        vq_resp = self.chat_model.generate_content(
            f"Given this answer:\n{initial_answer}\n\n"
            "List up to 3 short factual questions that would verify the key claims. "
            "Output only the questions, one per line, no numbering or bullets."
        ).text.strip()
        verification_questions = [q.strip() for q in vq_resp.splitlines() if q.strip()][:3]

        # Step 3: answer each verification question from context only
        contradictions = []
        verification_pairs = []

        for vq in verification_questions:
            vq_answer = self.chat_model.generate_content(
                f"Using only this context:\n{context_text}\n\n"
                f"Answer in one sentence: {vq}\n"
                "If the context doesn't contain the answer, reply with exactly: NOT FOUND"
            ).text.strip()

            verification_pairs.append({"question": vq, "answer": vq_answer})

            if "NOT FOUND" in vq_answer.upper():
                continue

            # Step 4: contradiction check
            verdict = self.chat_model.generate_content(
                f"Original answer excerpt:\n{initial_answer[:600]}\n\n"
                f"Verification question: {vq}\n"
                f"Verification answer: {vq_answer}\n\n"
                "Does the verification answer contradict the original answer? "
                "Reply with YES or NO only."
            ).text.strip().upper()

            if verdict.startswith("YES"):
                contradictions.append({"question": vq, "conflicting_answer": vq_answer})

        return {
            "answer": initial_answer,
            "verified": len(contradictions) == 0,
            "contradictions": contradictions,
            "verification_pairs": verification_pairs,
        }

    def recreate_collection(self):
        self.store.reset()
        self.latest_source = None
        self._save_meta()
