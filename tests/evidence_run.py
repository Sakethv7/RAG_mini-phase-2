# evidence_run.py — run the SAME questions through the old path (rag.ask) and the new
# path (rag.ask_guarded, local guard + route) and record what happened. Counts real
# Gemini calls by wrapping the SDK. Run from the repo root with Ollama up:
#   LOCAL_LLM_MODEL=qwen3:8b python tests/evidence_run.py
# Questions are the first N of each kind in tests/labels.jsonl (file order, no cherry-picking).
# Writes docs/evidence/run.json.

import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# BACKEND=jev runs the guarded path on Jev and skips the old path (reuse run.json for it).
BACKEND = os.getenv("BACKEND", "local")
os.environ.update({"GUARD_BACKEND": BACKEND, "ROUTE_BACKEND": BACKEND, "RERANK_BACKEND": "off",
                   "DECISION_LOG": os.path.join(ROOT, "logs", "evidence_decisions.jsonl")})
os.environ.setdefault("LOCAL_LLM_MODEL", "qwen3:8b")

import google.generativeai as genai  # noqa: E402
from google.api_core.exceptions import ResourceExhausted  # noqa: E402

import decisions  # noqa: E402
from rag_simple import SimpleRAG  # noqa: E402

N = {"attack": 4, "off_topic": 3, "chat": 3, "lookup": 3}
count = {"chat_ok": 0, "chat_429": 0, "embed": 0}

_gen = genai.GenerativeModel.generate_content
_emb = genai.embed_content


def counted_gen(self, *a, **k):
    try:
        out = _gen(self, *a, **k)
    except ResourceExhausted:
        count["chat_429"] += 1
        raise
    count["chat_ok"] += 1
    return out


def counted_emb(*a, **k):
    count["embed"] += 1
    return _emb(*a, **k)


genai.GenerativeModel.generate_content = counted_gen
genai.embed_content = counted_emb


def pick() -> list:
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "tests", "labels.jsonl"))]
    guard = [r for r in rows if r["step"] == "guard"]
    route = [r for r in rows if r["step"] == "route"]
    sets = {"attack": [r for r in guard if r["label"] == "INJECTION"],
            "off_topic": [r for r in guard if r["label"] == "OFF_TOPIC"],
            "chat": [r for r in route if r["label"] == "CHAT"],
            "lookup": [r for r in route if r["label"] == "LOOKUP"]}
    return [(kind, r["question"]) for kind, rs in sets.items() for r in rs[:N[kind]]]


def run(fn, question):
    """Call fn(question). Retry on the free-tier 429. Report the last good attempt."""
    for _ in range(8):
        before = dict(count)
        start = time.monotonic()
        try:
            out = fn(question)
        except ResourceExhausted:
            time.sleep(20)
            continue
        except decisions.GuardUnavailable as e:
            return {"error": f"guard unavailable: {e}"}
        return {"out": out, "seconds": round(time.monotonic() - start, 2),
                "gemini_chat_calls": count["chat_ok"] - before["chat_ok"],
                "gemini_embed_calls": count["embed"] - before["embed"]}
    return {"error": "quota never cleared"}


def main():
    rag = SimpleRAG()
    if BACKEND == "local":
        decisions.local_llm("warm up")  # load the model once, keep it out of the timings
    rows = []
    for kind, q in pick():
        print(kind, "|", q, flush=True)
        old = None if BACKEND == "jev" else run(rag.ask, q)
        new = run(rag.ask_guarded, q)
        rows.append({"kind": kind, "question": q, "old_ask": old, "guarded": new})
    out = {"date": datetime.now(timezone.utc).isoformat(), "chat_model": rag.chat_model_name,
           "local_model": os.environ["LOCAL_LLM_MODEL"] + " via Ollama" if BACKEND == "local" else "jev",
           "backends": decisions.backends(), "rows": rows, "totals": count}
    with open(os.path.join(ROOT, "docs", "evidence", "run.json" if BACKEND == "local" else f"run_{BACKEND}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["totals"]))


if __name__ == "__main__":
    main()
