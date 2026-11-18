# eval_decisions.py — compare the llm backend (Gemini) with the local backend (Qwen3 via
# Ollama) on the labeled set in tests/labels.jsonl. Run from the repo root with
# Ollama up, LOCAL_LLM_MODEL=qwen3:8b and GOOGLE_API_KEY set:
#   python tests/eval_decisions.py
# Writes docs/system1_results.json and prints it.

import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import google.generativeai as genai  # noqa: E402
from google.api_core.exceptions import ResourceExhausted  # noqa: E402

import decisions  # noqa: E402
from rag_simple import SimpleRAG  # noqa: E402

os.environ["DECISION_LOG"] = os.path.join(ROOT, "logs", "eval_decisions.jsonl")  # keep eval out of the prod log


class Llm:
    """Gemini call with retry on the free-tier 429. Records the time of the last good call."""

    def __init__(self, model_name: str):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.calls, self.chars, self.last_ms = 0, 0, 0

    def reset(self):
        self.calls, self.chars = 0, 0

    def __call__(self, prompt: str) -> str:
        for _ in range(10):
            start = time.monotonic()
            try:
                text = self.model.generate_content(prompt).text
            except ResourceExhausted:
                time.sleep(20)
                continue
            self.last_ms = int((time.monotonic() - start) * 1000)
            self.calls += 1
            self.chars += len(prompt)
            return text
        raise RuntimeError("quota never cleared")


class LocalCounter:
    """Wraps decisions.local_llm to count calls and prompt size (same fields as Llm)."""

    def __init__(self):
        self.calls, self.chars, self._real = 0, 0, decisions.local_llm
        decisions.local_llm = self

    def reset(self):
        self.calls, self.chars = 0, 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.chars += len(prompt)
        return self._real(prompt)


def server_rss_mb() -> float:
    pid = subprocess.run(["pgrep", "-f", "ollama runner"], capture_output=True, text=True).stdout.strip()
    if not pid:
        return 0.0
    return int(subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout) / 1024


def pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def run_classifier(step: str, backend: str, items: list, llm: Llm, local: LocalCounter) -> dict:
    os.environ[f"{step.upper()}_BACKEND"] = backend
    fn = getattr(decisions, step)
    preds, lat = [], []
    counter = local if backend == "local" else llm
    counter.reset()
    for it in items:
        try:
            d = fn(it["question"], llm)
            preds.append(d.label)
            lat.append(llm.last_ms if backend == "llm" else d.latency_ms)
        except decisions.GuardUnavailable:
            preds.append("ERROR")
            lat.append(0)
    golds = [it["label"] for it in items]
    recall = defaultdict(lambda: [0, 0])
    for g, p in zip(golds, preds):
        recall[g][1] += 1
        recall[g][0] += g == p
    return {
        "preds": preds,
        "accuracy": sum(g == p for g, p in zip(golds, preds)) / len(items),
        "recall": {k: v[0] / v[1] for k, v in recall.items()},
        "median_ms": statistics.median(lat),
        "p95_ms": pct(lat, 0.95),
        "model_calls_per_q": counter.calls / len(items),
        "input_tokens_per_q": counter.chars / 4 / len(items),  # rough: 4 characters per token
        "errors": preds.count("ERROR"),
    }


def norm(t: str) -> str:
    return re.sub(r"\s+", "", t.lower())


def run_rerank(backend: str, items: list, cache: dict, llm: Llm, local: LocalCounter) -> dict:
    os.environ["RERANK_BACKEND"] = backend
    counter = local if backend == "local" else llm
    counter.reset()
    ranks, hit5, p5, lat, fallbacks = [], [], [], [], 0
    for it in items:
        chunks = cache[it["question"]]
        order, d = decisions.rerank(it["question"], chunks, len(chunks), llm)
        rel = [any(s in norm(c["text"]) for s in it["answer_contains"]) for c in order]
        first = next((i + 1 for i, r in enumerate(rel) if r), None)
        ranks.append(1 / first if first else 0.0)
        hit5.append(any(rel[:5]))
        p5.append(sum(rel[:5]) / 5)
        lat.append(llm.last_ms if backend == "llm" else d.latency_ms)
        fallbacks += d.fallback
    return {
        "mrr": statistics.mean(ranks), "hit_at_5": statistics.mean(hit5), "precision_at_5": statistics.mean(p5),
        "median_ms": statistics.median(lat), "fallbacks": fallbacks,
        "model_calls_per_q": counter.calls / len(items), "input_tokens_per_q": counter.chars / 4 / len(items),
    }


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "tests", "labels.jsonl"))]
    by_step = {s: [r for r in rows if r["step"] == s] for s in ("guard", "route", "rerank")}
    rag = SimpleRAG()
    llm = Llm(rag.chat_model_name)

    local = LocalCounter()
    decisions.local_llm("warm up")  # first call loads the prompt cache; keep it out of the timings
    cache_note = server_rss_mb()

    cache = {it["question"]: rag.retrieve(it["question"], k=20) for it in by_step["rerank"]}

    out = {"n": {s: len(v) for s, v in by_step.items()}, "chat_model": rag.chat_model_name,
           "local_model": "qwen3:8b via Ollama", "server_rss_mb_after_warmup": cache_note}
    for step in ("guard", "route"):
        out[step] = {b: run_classifier(step, b, by_step[step], llm, local) for b in ("local", "llm")}
        agree = sum(a == b for a, b in zip(out[step]["local"]["preds"], out[step]["llm"]["preds"]))
        out[step]["agreement"] = agree / len(by_step[step])
    out["rerank"] = {b: run_rerank(b, by_step["rerank"], cache, llm, local) for b in ("off", "local", "llm")}

    out["server_rss_mb_end"] = server_rss_mb()
    with open(os.path.join(ROOT, "docs", "system1_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
