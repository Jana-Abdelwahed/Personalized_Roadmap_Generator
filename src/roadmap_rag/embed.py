import os
import hashlib
from typing import List
import numpy as np

def _hash_to_vec(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic offline embedding (fallback). NOT semantic, just stable."""
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    repeated = (h * ((dim // len(h)) + 1))[:dim]
    vec = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm

def openai_embed(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed with OpenAI if OPENAI_API_KEY is set; otherwise fallback to hash embeddings."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return np.stack([_hash_to_vec(t) for t in texts], axis=0)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(input=texts, model=model)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        vecs = [v / (np.linalg.norm(v) + 1e-8) for v in vecs]
        return np.stack(vecs, axis=0)
    except Exception:
        return np.stack([_hash_to_vec(t) for t in texts], axis=0)
