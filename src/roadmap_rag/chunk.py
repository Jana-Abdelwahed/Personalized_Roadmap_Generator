from typing import List, Tuple

def chunk_text(doc_id: str, text: str, chunk_size: int = 800, overlap: int = 120) -> List[Tuple[str, str]]:
    """Simple sliding window chunking. Returns list of (chunk_id, chunk_text)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk_id = f"{doc_id}#chunk-{idx}"
        chunks.append((chunk_id, " ".join(chunk_words)))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
        idx += 1
    return chunks
