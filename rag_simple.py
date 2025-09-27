# rag_simple.py — Docker-free RAG with a local NumPy vector store + Gemini
import os
import re
import glob
import uuid
import json
import hashlib
from typing import List, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import google.generativeai as genai


# ---------------------------
# helpers
# ---------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Simple sliding-window chunker:
      - chunk_size: characters per chunk
      - overlap:    chars carried into the next chunk
    """
    text = text.strip()
    if not text:
        return []
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + chunk_size, n)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out


def _point_id(source: str, chunk_id: int) -> int:
    """
    Deterministic 64-bit *signed* ID for (source, chunk_id);
    using signed avoids OverflowError when casting to np.int64.
    """
    b = hashlib.blake2b(f"{source}:{chunk_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(b, "big", signed=True)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_md(path: str) -> str:
    text = _read_txt(path)
    # light cleanup: strip code fences, convert [text](url) -> text, drop leading '#'
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    return text


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def _trim(s: str, max_chars: int) -> str:
    """Trim string to at most max_chars (adds … if trimmed)."""
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "…"


def _first_sentences(text: str, n: int = 2) -> str:
    """Quick & simple sentence splitter for extractive fallback."""
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def _bullets_from_text(text: str, n: int = 5) -> str:
    """Extractive fallback: turn first sentences into bullets."""
    parts = [p.strip() for p in re.split(r"(?<=[.?!])\s+", text.strip()) if p.strip()]
    picks = [f"• {p[:120]}" for p in parts[:n]]
    return "\n".join(picks) if picks else ""


def _normalize_query_text(q: str) -> str:
    """Normalize common misspellings/ASCII variants to improve retrieval."""
    repl = {
        "baharata": "bharata",
        "dasrjana": "dasarajna",
        "dasharajna": "dasarajna",
        "parushni": "paruṣṇī",
        "parusni": "paruṣṇī",
        "viswamitra": "viśvāmitra",
        "vasistha": "vasiṣṭha",
        "sudasa": "sudās",
    }
    q2 = q.lower()
    for k, v in repl.items():
        q2 = q2.replace(k, v)
    return q2


# ---------------------------
# Tiny on-disk vector store (NumPy)
# ---------------------------

class LiteVectorStore:
    """
    Minimal cosine-sim vector index persisted to disk.
    Files:
      - vectors.npy    -> (N, D) float32, L2-normalized
      - ids.npy        -> (N,) int64 stable ids
      - payloads.json  -> list[dict] with source, chunk_id, text, doc_id
    """
    def __init__(self, index_dir: str = "./lite_index"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.vectors: Optional[np.ndarray] = None
        self.ids: Optional[np.ndarray] = None
        self.payloads: List[Dict] = []
        self._load()

    # ---- paths ----
    def _vec_path(self): return os.path.join(self.index_dir, "vectors.npy")
    def _ids_path(self): return os.path.join(self.index_dir, "ids.npy")
    def _pl_path(self):  return os.path.join(self.index_dir, "payloads.json")

    # ---- load/save ----
    def _load(self):
        if os.path.exists(self._vec_path()) and os.path.exists(self._ids_path()) and os.path.exists(self._pl_path()):
            try:
                self.vectors = np.load(self._vec_path())
                self.ids = np.load(self._ids_path())
                with open(self._pl_path(), "r", encoding="utf-8") as f:
                    self.payloads = json.load(f)
            except Exception:
                # corrupted index -> reset
                self.vectors, self.ids, self.payloads = None, None, []
        else:
            self.vectors, self.ids, self.payloads = None, None, []

    def _save(self):
        if self.vectors is None:
            # empty index
            if os.path.exists(self._vec_path()): os.remove(self._vec_path())
            if os.path.exists(self._ids_path()): os.remove(self._ids_path())
            with open(self._pl_path(), "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False)
            return
        np.save(self._vec_path(), self.vectors.astype(np.float32))
        np.save(self._ids_path(), self.ids.astype(np.int64))
        with open(self._pl_path(), "w", encoding="utf-8") as f:
            json.dump(self.payloads, f, ensure_ascii=False)

    # ---- upsert ----
    def upsert(self, new_ids: np.ndarray, new_vecs: np.ndarray, new_payloads: List[Dict]):
        """
        new_ids: (M,) int64
        new_vecs: (M, D) float32, normalized
        new_payloads: list of length M
        """
        assert new_vecs.ndim == 2 and new_vecs.shape[0] == new_ids.shape[0] == len(new_payloads)
        if self.vectors is None:
            self.ids = new_ids.copy()
            self.vectors = new_vecs.copy()
            self.payloads = list(new_payloads)
            self._save()
            return

        # build id -> row map
        row = {int(i): idx for idx, i in enumerate(self.ids.tolist())}
        to_update = []
        to_add = []
        for j, pid in enumerate(new_ids.tolist()):
            if pid in row:
                to_update.append((row[pid], j))
            else:
                to_add.append(j)

        # update rows
        for i_old, j_new in to_update:
            self.vectors[i_old] = new_vecs[j_new]
            self.payloads[i_old] = new_payloads[j_new]

        # append new rows
        if to_add:
            self.ids = np.concatenate([self.ids, new_ids[to_add]])
            self.vectors = np.vstack([self.vectors, new_vecs[to_add]])
            self.payloads.extend([new_payloads[j] for j in to_add])

        self._save()

    # ---- search ----
    def search(self, qvec: np.ndarray, k: int = 5) -> List[Dict]:
        """
        qvec: (D,) normalized. Returns top-k payloads with 'score'.
        """
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []
        # cosine similarity for normalized vectors -> dot product
        sims = np.dot(self.vectors, qvec.astype(np.float32))
        k = max(1, min(k, sims.shape[0]))
        idxs = np.argpartition(-sims, kth=k-1)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]  # sort top-k
        out = []
        for i in idxs:
            item = dict(self.payloads[i])  # copy
            item["score"] = float(sims[i])
            out.append(item)
        return out

    # ---- maintenance ----
    def reset(self):
        self.vectors, self.ids, self.payloads = None, None, []
        self._save()


# ---------------------------
# RAG (no external DB)
# ---------------------------

class SimpleRAG:
    def __init__(
        self,
        collection_name: str = "mini_docs",                 # kept for compatibility; unused
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: str = "./lite_index",
        **kwargs,  # ignore any docker/qdrant args passed by older scripts
    ):
        """
        Minimal, framework-free RAG:
          - Sentence-Transformers (local) for embeddings
          - LiteVectorStore (NumPy) for vector search (on disk)
          - Google AI Studio (Gemini) for final answer generation
        """
        load_dotenv()

        # 1) Embeddings (local)
        self.embedder = SentenceTransformer(embedding_model)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # 2) Local vector store
        self.store = LiteVectorStore(index_dir=index_dir)

        # 3) Gemini (LLM)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY missing in .env")
        genai.configure(api_key=api_key)

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=(
                "Use ONLY the provided context. If the answer isn't fully in context, say you don't know. "
                "Keep it concise. Cite sources inline like [source chunk]."
            ),
        )

        # Larger answer budget + prompt budget to avoid truncation
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_output_tokens": 640,
        }
        self.max_prompt_chars = 12000

    # ---- Embedding helpers ----
    def embed(self, texts: List[str]) -> np.ndarray:
        # normalize for cosine distance
        return self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    # ---- Ingestion ----
    def _load_file_text(self, path: str) -> str:
        ext = os.path.splitext(path.lower())[1]
        if ext == ".txt":
            return _read_txt(path)
        if ext == ".md":
            return _read_md(path)
        if ext == ".pdf":
            return _read_pdf(path)
        return _read_txt(path)  # fallback

    def ingest_folder(self, folder: str = "documents") -> Dict:
        """Read .txt/.md/.pdf, chunk, embed, and upsert to local store."""
        patterns = ["*.txt", "*.md", "*.pdf"]
        paths = []
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(folder, pat)))
        paths = sorted(paths)

        total_chunks, details = 0, []

        for p in paths:
            try:
                text = self._load_file_text(p)
            except Exception as e:
                print(f"[skip] Failed reading {p}: {e}")
                continue

            chunks = chunk_text(text, chunk_size=800, overlap=200)
            if not chunks:
                continue

            vecs = self.embed(chunks)  # (M, D) normalized
            src_name = os.path.basename(p)
            doc_id = str(uuid.uuid4())

            ids = np.array([_point_id(src_name, i) for i in range(len(chunks))], dtype=np.int64)
            payloads = []
            for i, ch in enumerate(chunks):
                payloads.append({
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "text": ch,
                    "source": src_name,
                })

            self.store.upsert(ids, vecs, payloads)
            total_chunks += len(chunks)
            details.append({"file": src_name, "chunks": len(chunks)})

        return {"files_indexed": len(paths), "chunks_indexed": total_chunks, "details": details}

    # ---- Retrieval ----
    def retrieve(self, question: str, k: int = 5) -> List[Dict]:
        q_norm = _normalize_query_text(question)
        qv = self.embed([q_norm])[0]  # (D,) normalized
        hits = self.store.search(qv, k=k)
        # normalize the output fields to match the rest of the code
        out = []
        for h in hits:
            out.append({
                "score": h.get("score", 0.0),
                "source": h.get("source", ""),
                "chunk_id": h.get("chunk_id", -1),
                "text": h.get("text", ""),
            })
        return out

    # ---- build prompt with char budgeting ----
    def _build_prompt(self, question: str, contexts: List[Dict], max_chars: int) -> str:
        header = (
            "Use ONLY the provided context. If it's not fully in context, say you don't know. "
            "Respond as EXACTLY 5 bullet points, each ≤20 words. "
            "Cite like [source chunk].\n\n"
            f"Question:\n{question}\n\nContext:\n"
        )
        remaining = max_chars - len(header)
        parts = []
        for c in contexts:
            # keep chunks tidy; PDFs can be long
            txt = _trim(c["text"], 1200)
            part = f"[{c['source']} chunk {c['chunk_id']}]\n{txt}\n\n---\n\n"
            if len(part) <= remaining:
                parts.append(part)
                remaining -= len(part)
            else:
                break
        return header + "".join(parts)

    # ---- Answering (Gemini) ----
    def answer(self, question: str, k: int = 2) -> Dict:
        contexts = self.retrieve(question, k=k)

        if not contexts:
            return {
                "question": question,
                "matches": [],
                "response": "I don't have any indexed context. Add files to documents/ and run ingest.py."
            }

        # 1st attempt: k contexts (trimmed) within prompt budget
        prompt = self._build_prompt(question, contexts, self.max_prompt_chars)

        def _extract_text(resp):
            cands = getattr(resp, "candidates", None) or []
            for cand in cands:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content else []
                piece = "".join(getattr(p, "text", "") for p in parts)
                if piece:
                    return piece.strip(), getattr(cand, "finish_reason", None)
            return "", getattr(cands[0], "finish_reason", None) if cands else None

        try:
            resp = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
            )
            text, finish_reason = _extract_text(resp)
        except Exception as e2:
            text, finish_reason = "", f"EXC:{e2}"

        # Retry with fewer chunks if empty / MAX_TOKENS
        finish_name = getattr(finish_reason, "name", str(finish_reason)) if finish_reason is not None else None
        if (not text) or (finish_name == "MAX_TOKENS"):
            smaller = contexts[:1]  # just top-1 chunk
            prompt_small = self._build_prompt(question, smaller, self.max_prompt_chars // 2)
            try:
                resp2 = self.model.generate_content(
                    prompt_small,
                    generation_config=self.generation_config,
                )
                text2, _ = _extract_text(resp2)
                if text2:
                    text = text2
            except Exception:
                pass

        # FINAL GUARD: extractive fallback (no LLM) -> 5 bullets from top chunk
        if not text:
            top = contexts[0]
            text = _bullets_from_text(top["text"], n=5) or f"{_first_sentences(top['text'], 2)} [{top['source']} chunk {top['chunk_id']}]"

        return {"question": question, "matches": contexts, "response": text}

    # ---- Maintenance ----
    def recreate_collection(self):
        """For compatibility with earlier examples; wipes the local index."""
        self.store.reset()
