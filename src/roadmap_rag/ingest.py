import os
import re
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def fetch_url(url: str, timeout: int = 20) -> str:
    """Fetch plain text from a URL using requests + BeautifulSoup. Returns empty string on failure."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "roadmap-rag/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator=" ")
        return clean_text(text)
    except Exception:
        return ""

def load_seed_directory(seed_dir: str) -> List[Tuple[str, str]]:
    """Load local markdown files as (doc_id, text)."""
    docs = []
    if not os.path.isdir(seed_dir):
        return docs
    for fname in os.listdir(seed_dir):
        p = os.path.join(seed_dir, fname)
        if os.path.isfile(p) and fname.lower().endswith((".md", ".txt")):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((fname, f.read()))
    return docs

def ingest(seed_urls: List[str], seed_dir: str) -> List[Tuple[str, str]]:
    """Fetch from URLs (best-effort) and also load local seed docs. Returns list of (doc_id, text)."""
    docs = []
    for u in seed_urls:
        txt = fetch_url(u)
        if txt:
            docs.append((u, txt))
    docs.extend(load_seed_directory(seed_dir))
    seen = set()
    unique = []
    for k, v in docs:
        if k not in seen and v.strip():
            unique.append((k, v))
            seen.add(k)
    return unique
