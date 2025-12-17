from typing import List
import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T

def top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 5) -> List[int]:
    sims = cosine_sim(query_vec, doc_vecs)  # (1, N)
    order = np.argsort(-sims[0])[:k]
    return order.tolist()
