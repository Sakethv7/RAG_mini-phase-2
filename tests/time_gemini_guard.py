# time_gemini_guard.py: time the Gemini guard call on the 60 hard questions, one run.
# Run from the repo root:   python3 tests/time_gemini_guard.py
# Needs GOOGLE_API_KEY in .env. Optional: GEMINI_MODEL (default models/gemini-3.5-flash-lite).
# Calls the Gemini REST API with httpx. No system prompt and no generation config, like the app.
# One warm-up call is made first and not counted. Writes docs/evidence/hard_llm_retime.json.

import json
import os
import statistics
import sys

import httpx

os.environ["BACKEND"] = "llm"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evidence_hard as eh  # noqa: E402  (loads .env, sets GUARD_BACKEND=llm, holds HARD)

MODEL = os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent"
KEY = os.environ["GOOGLE_API_KEY"]


def llm(prompt: str) -> str:
    r = httpx.post(URL, headers={"x-goog-api-key": KEY}, timeout=60,
                   json={"contents": [{"parts": [{"text": prompt}]}]})
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


try:
    eh.decisions.guard("hello", llm)  # warm-up: connection setup, not counted
except eh.decisions.GuardUnavailable as e:
    sys.exit(f"warm-up call failed: {e}")

rows = []
for gold, q in eh.HARD:
    try:
        d = eh.decisions.guard(q, llm)
        got, ms = d.label, d.latency_ms
    except eh.decisions.GuardUnavailable:
        got, ms = "ERROR", None
    rows.append({"gold": gold, "question": q, "got": got, "ms": ms})

times = sorted(r["ms"] for r in rows if r["ms"] is not None)
out = {"model": MODEL, "n": len(rows), "timed": len(times),
       "errors": sum(r["got"] == "ERROR" for r in rows),
       "wrong_label": sum(r["got"] != r["gold"] for r in rows),
       "valid_refused": sum(r["gold"] == "SAFE" and r["got"] not in ("SAFE", "ERROR") for r in rows),
       "let_through": sum(r["gold"] != "SAFE" and r["got"] == "SAFE" for r in rows),
       "median_ms": int(statistics.median(times)), "slowest_ms": times[-1], "rows": rows}
with open(os.path.join(eh.ROOT, "docs", "evidence", "hard_llm_retime.json"), "w") as f:
    json.dump(out, f, indent=2)
print({k: v for k, v in out.items() if k != "rows"})
