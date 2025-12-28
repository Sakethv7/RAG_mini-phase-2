# rag_simple.py — Local NumPy RAG + Gemini (Render-safe, ENV-driven)

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
from google.api_core import exceptions as google_exceptions
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


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
    def __init__(self, index_dir="./lite_index"):
        load_dotenv()

        # --- ENV ---
        index_dir = os.getenv("INDEX_DIR", index_dir)
        os.makedirs(index_dir, exist_ok=True)
        self.index_dir = index_dir

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        self.chat_model_name = os.getenv(
            "GEMINI_MODEL", "models/gemini-2.0-flash"
        )
        self.embed_provider = os.getenv("EMBED_PROVIDER", "gemini").lower()
        self.embed_model_name = os.getenv(
            "GEMINI_EMBED_MODEL", "models/embedding-001"
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
        self._load_meta()

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
                self.latest_source = json.load(f).get("latest_source")

    def _save_meta(self):
        with open(self.meta_path, "w") as f:
            json.dump({"latest_source": self.latest_source}, f)

    # -------- Embeddings --------

    def embed(self, texts: List[str]) -> np.ndarray:
        print(f"Embedding {len(texts)} chunks via {self.embed_provider}")

        if self.embed_provider == "gemini":
            vectors = []
            for i, t in enumerate(texts):
                try:
                    r = genai.embed_content(
                        model=self.embed_model_name,
                        content=t
                    )
                    vectors.append(r["embedding"])
                except Exception as e:
                    print(f"❌ Gemini embedding failed at chunk {i}: {e}")
                    raise
            return np.array(vectors, dtype=np.float32)

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

    def recreate_collection(self):
        self.store.reset()
        self.latest_source = None
        self._save_meta()

    def list_sources(self):
        if hasattr(self.store, "list_sources"):
            return self.store.list_sources()
        return []
