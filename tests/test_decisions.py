import json

import pytest

from decisions import GuardUnavailable, guard, route


@pytest.fixture(autouse=True)
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("DECISION_LOG", str(path))
    return path


def fake(reply):
    return lambda prompt: reply


def raises(prompt):
    raise TimeoutError("boom")


@pytest.mark.parametrize("reply,label", [
    ("SAFE", "SAFE"),
    ("injection.", "INJECTION"),
    ("**OFF_TOPIC**\nbecause it asks for code", "OFF_TOPIC"),
])
def test_guard_parses_labels(reply, label):
    assert guard("q", fake(reply)).label == label


@pytest.mark.parametrize("llm", [fake("maybe"), fake(""), raises])
def test_guard_fails_closed(llm):
    with pytest.raises(GuardUnavailable):
        guard("q", llm)


def test_route_parses_label():
    d = route("q", fake("summary"))
    assert (d.label, d.fallback) == ("SUMMARY", False)


@pytest.mark.parametrize("llm", [fake("???"), raises])
def test_route_falls_back_to_lookup(llm):
    d = route("q", llm)
    assert (d.label, d.fallback) == ("LOOKUP", True)


def test_every_call_is_logged(log_path):
    guard("hello", fake("SAFE"))
    route("hello", fake("CHAT"))
    lines = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert [(l["step"], l["label"]) for l in lines] == [("guard", "SAFE"), ("route", "CHAT")]
    assert lines[0]["question"] == "hello"


def test_log_failure_does_not_raise(monkeypatch):
    monkeypatch.setenv("DECISION_LOG", "/proc/forbidden/decisions.jsonl")
    assert guard("q", fake("SAFE")).label == "SAFE"


# ---- ADR-4: backend dispatch and rerank ----

from decisions import backends, rerank  # noqa: E402

CHUNKS = [{"text": f"passage {i}", "source": "a.txt", "chunk_id": i, "score": 0.9 - i * 0.1} for i in range(4)]


def test_backends_defaults(monkeypatch):
    for s in ("GUARD", "ROUTE", "RERANK"):
        monkeypatch.delenv(f"{s}_BACKEND", raising=False)
    assert backends() == {"guard": "llm", "route": "llm", "rerank": "off"}


@pytest.mark.parametrize("var,value", [("GUARD_BACKEND", "off"), ("ROUTE_BACKEND", "gpt"), ("RERANK_BACKEND", "")])
def test_backends_reject_bad_values(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(RuntimeError):
        backends()


def test_route_off_returns_lookup(monkeypatch):
    monkeypatch.setenv("ROUTE_BACKEND", "off")
    d = route("q", raises)
    assert (d.backend, d.label, d.fallback) == ("off", "LOOKUP", False)


def test_local_backend_uses_local_llm(monkeypatch):
    import decisions
    monkeypatch.setenv("GUARD_BACKEND", "local")
    monkeypatch.setattr(decisions, "local_llm", lambda prompt: "INJECTION")
    d = guard("q", raises)  # the Gemini llm must not be called
    assert (d.backend, d.label) == ("local", "INJECTION")


def test_local_server_down_fails_closed(monkeypatch):
    import decisions
    def down(prompt):
        raise ConnectionError("local model server not running")
    monkeypatch.setenv("GUARD_BACKEND", "local")
    monkeypatch.setattr(decisions, "local_llm", down)
    with pytest.raises(GuardUnavailable):
        guard("q", fake("SAFE"))


def test_rerank_off_keeps_cosine_order(monkeypatch):
    monkeypatch.setenv("RERANK_BACKEND", "off")
    kept, d = rerank("q", CHUNKS, 2, raises)
    assert [c["chunk_id"] for c in kept] == [0, 1] and d.backend == "off"


def test_rerank_llm_reorders_and_appends_missing(monkeypatch):
    monkeypatch.setenv("RERANK_BACKEND", "llm")
    kept, d = rerank("q", CHUNKS, 4, fake("3, 1, 3, 99"))
    assert [c["chunk_id"] for c in kept] == [2, 0, 1, 3] and not d.fallback


@pytest.mark.parametrize("llm", [fake("no idea"), raises])
def test_rerank_llm_falls_back_to_cosine(monkeypatch, llm):
    monkeypatch.setenv("RERANK_BACKEND", "llm")
    kept, d = rerank("q", CHUNKS, 2, llm)
    assert [c["chunk_id"] for c in kept] == [0, 1] and d.fallback


def test_rerank_local_uses_local_llm(monkeypatch):
    import decisions
    monkeypatch.setenv("RERANK_BACKEND", "local")
    monkeypatch.setattr(decisions, "local_llm", lambda prompt: "2, 4")
    kept, d = rerank("q", CHUNKS, 2, raises)
    assert [c["chunk_id"] for c in kept] == [1, 3] and d.backend == "local" and not d.fallback
