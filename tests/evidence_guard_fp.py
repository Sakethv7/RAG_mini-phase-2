# evidence_guard_fp.py — send every benign question in tests/labels.jsonl (the 30 route
# questions: lookups, summaries, greetings) through the local guard. A correct guard says SAFE.
# Anything else is a false refusal. Run from the repo root with Ollama up:
#   LOCAL_LLM_MODEL=qwen3:8b python tests/evidence_guard_fp.py
# Writes docs/evidence/guard_false_refusals.json.

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.update({"GUARD_BACKEND": "local", "DECISION_LOG": os.path.join(ROOT, "logs", "evidence_decisions.jsonl")})
os.environ.setdefault("LOCAL_LLM_MODEL", "qwen3:8b")

import decisions  # noqa: E402


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "tests", "labels.jsonl"))]
    benign = [r for r in rows if r["step"] == "route"]
    decisions.local_llm("warm up")
    out = []
    for r in benign:
        d = decisions.guard(r["question"], None)
        out.append({"question": r["question"], "route_label": r["label"], "guard_label": d.label})
    bad = [o for o in out if o["guard_label"] != "SAFE"]
    summary = {"n": len(out), "false_refusals": len(bad), "items": out}
    with open(os.path.join(ROOT, "docs", "evidence", "guard_false_refusals.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(len(bad), "of", len(out), "benign questions refused")
    for o in bad:
        print(" ", o["guard_label"], "|", o["question"])


if __name__ == "__main__":
    main()
