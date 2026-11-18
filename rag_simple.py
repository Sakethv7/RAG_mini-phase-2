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
from google.api_core import exceptions as google_exceptions
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

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
    # Use unsigned IDs so backends like Qdrant accept them.
    h = hashlib.blake2b(f"{source}:{chunk_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms


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
        if self.vectors is None or self.vectors.size == 0:
            return []

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

    def list_sources(self):
        if not self.payloads:
            return []
        seen = {}
        for p in self.payloads:
            src = p.get("source")
            if not src:
                continue
            seen.setdefault(src, {"source": src, "chunks": 0, "latest_timestamp": ""})
            seen[src]["chunks"] += 1
            ts = p.get("timestamp") or ""
            if ts > seen[src]["latest_timestamp"]:
                seen[src]["latest_timestamp"] = ts
        return sorted(seen.values(), key=lambda x: x["source"])


class QdrantVectorStore:
    def __init__(self, url, api_key, collection):
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vectors_size = None
        self._read_existing_size()

    def _read_existing_size(self):
        """After a restart, learn the size of a collection that already exists. Without this,
        search and list_sources return nothing until the first upload of the session."""
        try:
            vectors = self.client.get_collection(self.collection).config.params.vectors
        except Exception:
            return  # no collection yet, or Qdrant unreachable: upsert creates it later
        if isinstance(vectors, qmodels.VectorParams):
            self.vectors_size = vectors.size

    def _ensure_collection(self, vector_size):
        if self.vectors_size == vector_size:
            return
        self.vectors_size = vector_size
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        vec_size = vectors.shape[1]
        self._ensure_collection(vec_size)
        points = []
        for i, v in enumerate(vectors):
            points.append(
                qmodels.PointStruct(
                    id=int(ids[i]),
                    vector=v.tolist(),
                    payload=payloads[i]
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, qvec, k=5, source_filter=None):
        if self.vectors_size is None:
            return []
        must = []
        if source_filter:
            must.append(qmodels.FieldCondition(
                key="source",
                match=qmodels.MatchValue(value=source_filter)
            ))
        if hasattr(self.client, "search"):
            res = self.client.search(
                collection_name=self.collection,
                query_vector=qvec.tolist(),
                limit=k,
                query_filter=qmodels.Filter(must=must) if must else None
            )
        elif hasattr(self.client, "search_points"):
            res = self.client.search_points(
                collection_name=self.collection,
                query_vector=qvec.tolist(),
                limit=k,
                query_filter=qmodels.Filter(must=must) if must else None
            )
        else:
            # Fallback to HTTP points_api if client is very old
            http_client = getattr(self.client, "http", None)
            if http_client and hasattr(http_client, "points_api"):
                points_api = http_client.points_api
                if hasattr(points_api, "search_points"):
                    res = points_api.search_points(
                        collection_name=self.collection,
                        search_request=qmodels.SearchRequest(
                            vector=qvec.tolist(),
                            limit=k,
                            filter=qmodels.Filter(must=must) if must else None,
                        ),
                    ).result
                elif hasattr(points_api, "search"):
                    res = points_api.search(
                        collection_name=self.collection,
                        search_request=qmodels.SearchRequest(
                            vector=qvec.tolist(),
                            limit=k,
                            filter=qmodels.Filter(must=must) if must else None,
                        ),
                    ).result
                else:
                    raise RuntimeError("qdrant-client points_api missing search; please upgrade qdrant-client")
            else:
                raise RuntimeError("qdrant-client is missing search; upgrade qdrant-client>=1.9.0")
        out = []
        for r in res:
            payload = r.payload or {}
            out.append({**payload, "score": float(r.score)})
        return out

    def reset(self):
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self.vectors_size = None

    def list_sources(self):
        if self.vectors_size is None:
            return []
        page = None
        seen = {}
        while True:
            scroll_res, page = self.client.scroll(
                collection_name=self.collection,
                with_payload=True,
                limit=128,
                offset=page
            )
            for p in scroll_res:
                payload = p.payload or {}
                src = payload.get("source")
                if not src:
                    continue
                seen.setdefault(src, {"source": src, "chunks": 0, "latest_timestamp": ""})
                seen[src]["chunks"] += 1
                ts = payload.get("timestamp") or ""
                if ts > seen[src]["latest_timestamp"]:
                    seen[src]["latest_timestamp"] = ts
            if not page:
                break
        return sorted(seen.values(), key=lambda x: x["source"])


# =========================
# SimpleRAG (Gemini)
# =========================

class SimpleRAG:
    def __init__(self, index_dir="./lite_index", check_index=True):
        load_dotenv()

        # --- ENV ---
        index_dir = os.getenv("INDEX_DIR", index_dir)
        os.makedirs(index_dir, exist_ok=True)
        self.index_dir = index_dir

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        self.chat_model_name = os.getenv(
            "GEMINI_MODEL", "models/gemini-3.5-flash-lite"
        )
        # Embedding provider: "gemini" (gemini-embedding-001, truncated to GEMINI_EMBED_DIM and
        # L2-normalized so dot product == cosine similarity) or "local" (MiniLM, 384 dims).
        self.embed_provider = os.getenv("EMBED_PROVIDER", "gemini").lower()
        self.embed_model_name = os.getenv(
            "GEMINI_EMBED_MODEL", "models/gemini-embedding-001"
        )
        self.local_embed_model = os.getenv(
            "LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", 8))
        self.answer_top_n = int(os.getenv("ANSWER_TOP_N", 5))
        self.vector_backend = os.getenv("VECTOR_BACKEND", "local").lower()
        self.qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
        self.qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip()
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "rag_docs").strip()
        # What the index must match: model name and dimension of the active embedder.
        # Local MiniLM dimension is not fixed by this code, so it is not checked (None).
        if self.embed_provider == "gemini":
            self.index_embed_model = self.embed_model_name
            self.embed_dim = int(os.getenv("GEMINI_EMBED_DIM", "768"))
        else:
            self.index_embed_model = self.local_embed_model
            self.embed_dim = None

        genai.configure(api_key=api_key)

        if self.vector_backend == "qdrant":
            if not (self.qdrant_url and self.qdrant_api_key):
                raise RuntimeError("VECTOR_BACKEND=qdrant requires QDRANT_URL and QDRANT_API_KEY")
            self.store = QdrantVectorStore(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection=self.qdrant_collection,
            )
        else:
            self.store = LiteVectorStore(index_dir)
        self.meta_path = f"{self.index_dir}/meta.json"

        self.latest_source = None
        self.meta_embed_model = None
        backends()  # fail at startup on a bad *_BACKEND value
        self._load_meta()
        if check_index:
            self._check_index_matches_embedder()

        self._local_encoder = None

        self.chat_model = genai.GenerativeModel(
            model_name=self.chat_model_name,
            system_instruction=(
                "You are a document analysis assistant.\n"
                "Use ONLY the provided context chunks.\n"
                "Be concise and helpful. Provide summaries or answers based on the question.\n"
                "If asked for playful or creative phrasing, you may add light, non-abusive humor; never be insulting or harmful.\n"
                "Avoid speculation; if information is missing, say so clearly."
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
                    "embed_model": self.index_embed_model,
                    "embed_dim": self.embed_dim,
                },
                f,
            )

    def _check_index_matches_embedder(self):
        v = getattr(self.store, "vectors", None)  # Qdrant has no local vectors to check
        if v is None:
            return
        dim_bad = self.embed_dim is not None and v.shape[1] != self.embed_dim
        if dim_bad or self.meta_embed_model != self.index_embed_model:
            raise RuntimeError(
                f"Index in {self.store.index_dir} has {v.shape[1]}-dim vectors "
                f"(model: {self.meta_embed_model or 'unknown'}), but the embedder is "
                f"{self.index_embed_model} at {self.embed_dim or 'unchecked'} dims. "
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
        print(f"Embedding {len(texts)} chunks via {self.embed_provider}")

        if self.embed_provider == "gemini":
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

        # Local embeddings
        try:
            if self._local_encoder is None:
                from sentence_transformers import SentenceTransformer

                self._local_encoder = SentenceTransformer(self.local_embed_model)
            vecs = self._local_encoder.encode(
                texts, convert_to_numpy=True, normalize_embeddings=False
            )
            return vecs.astype(np.float32)
        except ImportError as e:
            raise RuntimeError(
                "Local embedding model requested but sentence-transformers is not installed. "
                "pip install sentence-transformers"
            ) from e
        except Exception as e:
            print(f"❌ Local embedding failed: {e}")
            raise

    # -------- Ingestion --------

    def ingest_text(self, text: str, source: str):
        text = normalize_ocr_text(text)
        chunks = chunk_text(text)

        vectors = self.embed(chunks)
        vectors = _normalize_rows(vectors)
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

    def retrieve(self, question, k=None):
        if hasattr(self.store, "vectors"):
            if self.store.vectors is None or self.store.vectors.size == 0:
                return []

        qv = self.embed([question])[0]
        qv = qv / (np.linalg.norm(qv) or 1)
        return self.store.search(qv, k=k or self.retrieval_k)

    # -------- Answer --------

    def ask(self, question):
        contexts = self.retrieve(question, k=self.retrieval_k)
        contexts = contexts[: self.answer_top_n]

        if not contexts:
            return "No relevant information found in the indexed documents."

        context_text = "\n\n".join(
            f"[{c['source']}:{c['chunk_id']}]\n{c['text']}" for c in contexts
        )

        prompt = f"""
Question:
{question}

Context:
{context_text}

Formatting rules (must follow):
- Use Markdown where helpful; bullets are fine but not required.
- Keep it concise and relevant to the question.
- Be respectful; do not generate insults. If asked for humor, keep it light and non-abusive.
- If context is insufficient, say so explicitly. Avoid speculation.
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

    def list_sources(self):
        if hasattr(self.store, "list_sources"):
            return self.store.list_sources()
        return []
