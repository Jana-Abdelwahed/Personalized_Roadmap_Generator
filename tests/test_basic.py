import numpy as np
from roadmap_rag.chunk import chunk_text
from roadmap_rag.embed import openai_embed
from roadmap_rag.store import LocalStore
from roadmap_rag.retrieve import top_k

def test_chunking_basic():
    text = " ".join([f"w{i}" for i in range(1000)])
    chunks = chunk_text("doc1", text, chunk_size=200, overlap=50)
    assert len(chunks) > 0
    assert "doc1#chunk-0" in chunks[0][0]

def test_embedding_shape_offline(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    vecs = openai_embed(["a", "b", "c"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 3

def test_store_roundtrip(tmp_path):
    store = LocalStore(tmp_path.as_posix())
    chunks = [(f"id{i}", f"text {i}") for i in range(5)]
    vecs = np.eye(5, dtype=np.float32)
    store.add(chunks, vecs)
    store.save()

    other = LocalStore(tmp_path.as_posix())
    other.load()
    assert len(other) == 5
    assert other.chunks[0][1].startswith("text")

def test_retrieve_topk():
    doc_vecs = np.eye(5, dtype=np.float32)
    query = np.array([[1,0,0,0,0]], dtype=np.float32)
    idxs = top_k(query, doc_vecs, k=3)
    assert idxs[0] == 0
