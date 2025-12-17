import os
import json
from typing import List, Tuple
import numpy as np

class LocalStore:
    """Very lightweight local vector store: saves chunks + embeddings to a JSON + NPZ file."""
    def __init__(self, path: str):
        self.path = path
        self.meta_path = os.path.join(path, "index.json")
        self.vec_path = os.path.join(path, "index.npz")
        os.makedirs(path, exist_ok=True)
        self.chunks: List[Tuple[str, str]] = []
        self.vecs = None

    def add(self, chunks: List[Tuple[str, str]], vecs: np.ndarray):
        if len(chunks) != len(vecs):
            raise ValueError("Chunks and vectors must be same length")
        self.chunks.extend(chunks)
        if self.vecs is None:
            self.vecs = vecs
        else:
            self.vecs = np.vstack([self.vecs, vecs])

    def save(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f)
        if self.vecs is not None:
            np.savez(self.vec_path, vecs=self.vecs)

    def load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.chunks = [tuple(x) for x in json.load(f)]
        if os.path.exists(self.vec_path):
            self.vecs = np.load(self.vec_path)["vecs"]

    def __len__(self):
        return 0 if self.vecs is None else len(self.vecs)
