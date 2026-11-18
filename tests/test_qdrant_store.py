# A restarted app must find documents that are already in Qdrant. Uses Qdrant's in-memory mode.
import os
import sys

import numpy as np
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_simple import QdrantVectorStore  # noqa: E402


def make_store(client):
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.collection, store.client, store.vectors_size = "docs", client, None
    return store


def test_restart_finds_existing_collection():
    client = QdrantClient(":memory:")
    first = make_store(client)
    vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    first.upsert(np.array([1]), vec, [{"source": "a.pdf", "chunk_id": 0, "text": "hello"}])

    restarted = make_store(client)
    restarted._read_existing_size()
    assert restarted.vectors_size == 4
    assert [r["text"] for r in restarted.search(vec[0], k=1)] == ["hello"]
    assert restarted.list_sources()[0]["source"] == "a.pdf"


def test_missing_collection_stays_empty():
    store = make_store(QdrantClient(":memory:"))
    store._read_existing_size()
    assert store.vectors_size is None
    assert store.search(np.zeros(4, dtype=np.float32)) == []
