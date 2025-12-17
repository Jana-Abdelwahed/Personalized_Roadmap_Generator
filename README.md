# Personalized Roadmap RAG — Chat UI Edition

A ready-to-run, API-connected Streamlit RAG app with a **ChatGPT-like colorful UI**.
Works online with OpenAI (set `OPENAI_API_KEY`) and also offline (deterministic embeddings + local seed docs).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional: add your OpenAI key in .env (OPENAI_API_KEY=sk-...)
streamlit run src/app/streamlit_app.py
```

## Notes
- If you must write anything yourself: just set `OPENAI_API_KEY` in `.env`.
- Rebuild the index from the sidebar to ingest seed URLs + local seeds.
- Chat in the bottom input; messages render as colorful bubbles with avatars.
# personalized_roadmap_rag_chat_ready
