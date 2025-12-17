import os
from typing import List, Tuple
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from roadmap_rag.ingest import ingest
from roadmap_rag.chunk import chunk_text
from roadmap_rag.embed import openai_embed
from roadmap_rag.store import LocalStore
from roadmap_rag.retrieve import top_k
from roadmap_rag.generate import generate_answer
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Load .env (you only need to write your OPENAI_API_KEY here if you want real API generation) ---
load_dotenv(override=True)

# --- Page config ---
st.set_page_config(page_title="Personalized Roadmap RAG", page_icon="🗺️", layout="wide")

# --- Custom CSS for a ChatGPT-like colorful palette ---
st.markdown("""
<style>
:root {
  --bg: #0f1223;
  --panel: #151a2e;
  --panel-2: #1b2240;
  --text: #ecf0ff;
  --muted: #9aa3c7;
  --primary: #4f46e5;         
  --primary-2: #a78bfa;       
  --accent: #22d3ee;          
  --user: #3a86ff;            
  --assistant-1: #ff006e;     
  --assistant-2: #8338ec;     
  --success: #22c55e;
  --warn: #f59e0b;
  --error: #ef4444;
}

@keyframes floatIn {
  from { transform: translateY(6px); opacity: .0; }
  to { transform: translateY(0); opacity: 1; }
}

html, body, .stApp { background: radial-gradient(1200px 800px at 10% -10%, #1f2450 0%, var(--bg) 40%); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.stButton > button {
  background: linear-gradient(90deg, var(--primary) 0%, var(--primary-2) 100%);
  color: white;
  border: 0;
  border-radius: 12px;
  padding: 0.6rem 1rem;
  box-shadow: 0 8px 24px rgba(79,70,229,.35);
}
.stButton > button:hover { filter: brightness(1.06); }

.chat-wrap {
  background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.00) 100%);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 8px 16px;
}

.msg { display: flex; gap: 10px; align-items: flex-start; padding: 8px 10px; border-radius: 18px; animation: floatIn .18s ease-out both; margin-bottom: 8px; }
.msg .avatar { width: 32px; height: 32px; flex: 0 0 32px; border-radius: 50%; display: grid; place-items: center; font-size: 18px; color: white; }
.msg .body { flex: 1; line-height: 1.5; font-size: 0.98rem; padding: 10px 14px; border-radius: 16px; box-shadow: 0 6px 16px rgba(0,0,0,.25); }

.msg.user .avatar { background: var(--user); }
.msg.user .body { background: linear-gradient(135deg, #2f64ff 0%, #3a86ff 100%); color: white; }

.msg.assistant .avatar { background: linear-gradient(135deg, var(--assistant-1) 0%, var(--assistant-2) 100%); }
.msg.assistant .body { background: linear-gradient(135deg, rgba(255,0,110,.16) 0%, rgba(131,56,236,.16) 100%); border: 1px solid rgba(255,255,255,.10); }

.source { margin-top: 6px; font-size: 0.88rem; color: var(--muted); }
.source a { color: var(--accent); text-decoration: none; }
.source a:hover { text-decoration: underline; }

.stChatInputContainer textarea {
  background: var(--panel);
  color: var(--text);
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.08);
}

.status { padding: 8px 12px; border-radius: 10px; display: inline-flex; gap: 8px; align-items: center; border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.04); }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 999px; background: rgba(255,255,255,0.06); color: var(--text); font-size: 12px; }
.badge.green { background: rgba(34,197,94,.16); border: 1px solid rgba(34,197,94,.35); }
.badge.gray { background: rgba(148,163,184,.16); border: 1px solid rgba(148,163,184,.35); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    default_urls = os.getenv("SEED_URLS", "").replace(",", "\\n").strip()
    urls_str = st.text_area("Seed URLs (one per line)", value=default_urls, height=150,
                            help="These are used when you click Rebuild Index. Local docs in 'data/seed' are always added.")
    chunk_size = st.slider("Chunk size (words)", 200, 1200, 800, 50)
    overlap = st.slider("Overlap (words)", 0, 400, 120, 10)
    k = st.slider("Results k", 1, 10, 5, 1)
    embed_model = st.text_input("Embedding model", value="text-embedding-3-small")
    chat_model = st.text_input("Chat model", value="gpt-4o-mini")
    index_dir = st.text_input("Index directory", value=os.getenv("INDEX_DIR", "index"))
    seed_dir = st.text_input("Local seed dir", value=os.getenv("SEED_DIR", "data/seed"))
    rebuild = st.button("Rebuild Index", help="Fetch + chunk + (embed) to rebuild the local vector store.")

store = LocalStore(index_dir)

if rebuild:
    st.toast("Rebuilding index…", icon="🛠️")
    docs = ingest([u.strip() for u in urls_str.splitlines() if u.strip()], seed_dir)
    all_chunks: List[Tuple[str, str]] = []
    for doc_id, text in docs:
        all_chunks.extend(chunk_text(doc_id, text, chunk_size=chunk_size, overlap=overlap))
    if not all_chunks:
        st.error("No chunks produced. Check seed URLs or local seed dir.")
    else:
        vecs = openai_embed([c for _, c in all_chunks], model=embed_model)
        store = LocalStore(index_dir)
        store.add(all_chunks, vecs)
        store.save()
        st.success(f"Indexed {len(all_chunks)} chunks.")

store.load()

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"|"assistant", "content": str, "sources": List[str]}

col1, col2 = st.columns([0.72, 0.28])
with col1:
    st.title("🗺️ Personalized Roadmap RAG")
    st.caption("Chat-style RAG with colorful palette, API-connected (OpenAI) and offline fallbacks.")
with col2:
    api_on = bool(os.getenv("OPENAI_API_KEY", "").strip())
    st.markdown(
        f'<div class="status">'
        f'<span class="badge {"green" if api_on else "gray"}">API: {"ON" if api_on else "OFF"}</span>'
        f'<span class="badge gray">Index: {len(store)} vecs</span>'
        f'</div>', unsafe_allow_html=True
    )

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

def render_msg(role: str, content: str, sources: list | None = None):
    avatar = "🧑‍💻" if role == "user" else "🤖"
    cls = "user" if role == "user" else "assistant"
    body = content.replace("\\n", "<br/>")
    st.markdown(
        f"""
        <div class="msg {cls}">
          <div class="avatar">{avatar}</div>
          <div class="body">{body}
            {("" if not sources else '<div class="source">Sources: ' + " · ".join(sources) + "</div>")}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

for m in st.session_state.messages:
    render_msg(m["role"], m["content"], m.get("sources"))

prompt = st.chat_input("Ask anything about AI Engineering, RAG, evals, MLOps…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_msg("user", prompt)

    if len(store) == 0:
        answer = "Index is empty. Click **Rebuild Index** in the sidebar, then ask again."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        render_msg("assistant", answer)
    else:
        qvec = openai_embed([prompt], model=embed_model)
        idxs = top_k(qvec, store.vecs, k=k)
        context = [store.chunks[i] for i in idxs]

        src_labels = []
        for i, (cid, _) in enumerate(context):
            src = cid.split("#chunk-")[0]
            if src.startswith("http"):
                label = f"[{i}] " + (src[:60] + "…" if len(src) > 60 else src)
            else:
                label = f"[{i}] local:{src}"
            src_labels.append(label)

        answer = generate_answer(prompt, context, model=chat_model)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": src_labels})
        render_msg("assistant", answer, sources=src_labels)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Note: Set `OPENAI_API_KEY` in `.env` to enable real embeddings & generation. Otherwise the app falls back to offline deterministic embeddings and a templated answer that cites sources.")
