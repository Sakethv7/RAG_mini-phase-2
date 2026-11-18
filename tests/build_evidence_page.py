# build_evidence_page.py — turn docs/evidence/run.json and guard_false_refusals.json into one
# static HTML page (docs/visuals/evidence.html). No network, no libraries. Numbers come from
# the JSON files only. Run: python tests/build_evidence_page.py

import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run = json.load(open(os.path.join(ROOT, "docs", "evidence", "run.json")))
fp = json.load(open(os.path.join(ROOT, "docs", "evidence", "guard_false_refusals.json")))
rows = run["rows"]
e = html.escape

old_calls = sum(r["old_ask"]["gemini_chat_calls"] for r in rows)
new_calls = sum(r["guarded"]["gemini_chat_calls"] for r in rows)
refused = sum(1 for r in rows if r["guarded"]["out"]["refused"])
good_refusals = sum(1 for r in rows if r["guarded"]["out"]["refused"] and r["kind"] in ("attack", "off_topic"))
answered = [r for r in rows if not r["guarded"]["out"]["refused"]]
extra_s = sum(r["guarded"]["seconds"] - r["old_ask"]["seconds"] for r in answered) / len(answered)
fp_items = [i for i in fp["items"] if i["guard_label"] != "SAFE"]


def clip(t, n=150):
    return t if len(t) <= n else t[:n].rstrip() + "…"


def trace(r):
    return " → ".join(f'{t["step"]} {t["label"]} {t["latency_ms"]} ms' for t in r["guarded"]["out"]["trace"])


def tr(r):
    o, g = r["old_ask"], r["guarded"]["out"]
    bad = r["kind"] == "lookup" and g["refused"]
    cls = ' class="bad"' if bad else ""
    return (f'<tr{cls}><td>{e(r["kind"])}</td><td>{e(r["question"])}</td>'
            f'<td>{e(clip(o["out"]))}<br><small>{o["gemini_chat_calls"]} Gemini call, {o["seconds"]} s</small></td>'
            f'<td>{e(clip(g["answer"]))}<br><small>{e(trace(r))}<br>{r["guarded"]["gemini_chat_calls"]} Gemini call, '
            f'{r["guarded"]["seconds"]} s</small></td></tr>')


page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG_mini evidence</title>
<style>
:root{{--bg:#fff;--fg:#111827;--mut:#6b7280;--line:#e5e7eb;--card:#f3f4f6;--bad:#fecaca;--badfg:#450a0a;--acc:#1d4ed8}}
:root[data-theme=dark]{{--bg:#0f172a;--fg:#e5e7eb;--mut:#9ca3af;--line:#334155;--card:#1e293b;--bad:#7f1d1d;--badfg:#fee2e2;--acc:#93c5fd}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0f172a;--fg:#e5e7eb;--mut:#9ca3af;--line:#334155;--card:#1e293b;--bad:#7f1d1d;--badfg:#fee2e2;--acc:#93c5fd}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1100px;margin:auto}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}}
p.s{{color:var(--mut);margin:0 0 16px}} .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.tile{{background:var(--card);border-radius:10px;padding:14px}} .tile b{{font-size:26px;color:var(--acc);display:block}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}
th{{color:var(--mut);font-weight:600}} small{{color:var(--mut)}} tr.bad td{{background:var(--bad);color:var(--badfg)}} tr.bad small{{color:var(--badfg)}}
code{{background:var(--card);padding:1px 5px;border-radius:4px}} @media(max-width:700px){{td,th{{font-size:13px;padding:6px}}}}
</style></head><body><main>
<h1>RAG_mini: old path against guarded path</h1>
<p class="s">Measured {e(run["date"][:10])}. Chat model {e(run["chat_model"])}. Guard and route: {e(run["local_model"])}. Same {len(rows)} questions through both paths. Source: <code>docs/evidence/run.json</code>.</p>
<div class="tiles">
<div class="tile"><b>{old_calls} → {new_calls}</b>Gemini chat calls for {len(rows)} questions (old path, guarded path)</div>
<div class="tile"><b>{refused} of {len(rows)}</b>questions refused with no Gemini call. {good_refusals} were attacks or off-topic. {refused - good_refusals} were valid questions.</div>
<div class="tile"><b>+{extra_s:.1f} s</b>mean extra time on questions that still reach Gemini (local guard and route)</div>
<div class="tile"><b>{len(fp_items)} of {fp["n"]}</b>benign questions wrongly refused by the local guard</div>
</div>
<h2>Each question, both paths</h2>
<table><tr><th>Kind</th><th>Question</th><th>Old path (<code>/ask</code>)</th><th>Guarded path (<code>/ask/guarded</code>)</th></tr>
{"".join(tr(r) for r in rows)}</table>
<h2>False refusals on benign questions</h2>
<p class="s">All {fp["n"]} lookup, summary and greeting questions in the label set went through the local guard. A correct guard says SAFE for every one.</p>
<table><tr><th>Guard said</th><th>Question</th></tr>
{"".join(f'<tr class="bad"><td>{e(i["guard_label"])}</td><td>{e(i["question"])}</td></tr>' for i in fp_items) or '<tr><td>none</td><td>The guard said SAFE for all of them.</td></tr>'}</table>
<h2>Limits</h2>
<p class="s">Small samples. I picked the first questions of each kind in <code>tests/labels.jsonl</code>, in file order. I wrote those labels. I read the old-path answers myself. The free tier never returned a 429 in this run ({run["totals"]["chat_429"]} errors).</p>
</main>
<script>try{{var t=new URLSearchParams(location.search).get("theme");if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
</body></html>"""
out = os.path.join(ROOT, "docs", "visuals", "evidence.html")
open(out, "w").write(page)
print(out, old_calls, new_calls, refused, round(extra_s, 2), len(fp_items))
