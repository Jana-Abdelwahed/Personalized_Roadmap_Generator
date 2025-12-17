# 🗺️ Personalized Roadmap Generator (RAG)

An **AI-powered, fully offline-capable personalized learning roadmap generator** built using **Retrieval-Augmented Generation (RAG)**. The system creates **adaptive, up-to-date roadmaps** based on the user’s skill level and goals, avoiding outdated, one-size-fits-all roadmaps.

---

## 🚀 Features

* 🔍 **Retrieval-Augmented Generation (RAG)** for reliable, source-backed content
* 🧠 **Personalized roadmaps** (Beginner / Intermediate / Advanced)
* 📦 **Fully local & private** (Ollama + FAISS supported)
* ⚡ **Fast semantic search** using FAISS vector database
* 📝 **Editable exports**: Markdown / JSON (Notion & Obsidian friendly)
* 🌐 **Streamlit web interface** with chat-style UI
* 🔐 No cloud dependency → **zero cost & privacy-first**

---

## ❓ Why This Project?

Traditional learning roadmaps suffer from:

* ❌ Static & outdated content
* ❌ No personalization (everyone sees the same roadmap)
* ❌ Poor or abandoned learning resources
* ❌ Locked PDF formats

This project solves these issues by generating **dynamic, skill-aware roadmaps** using **live retrieval from curated sources**.

---

## 🏗️ Architecture Overview

```
User Query
   ↓
Streamlit UI
   ↓
RAG Pipeline (LangChain-style)
   ├── Ingestion (URLs + local docs)
   ├── Chunking
   ├── Embedding
   ├── FAISS Vector Store
   ├── Top-k Retrieval
   └── LLM Generation (Ollama / OpenAI)
   ↓
Personalized Roadmap (Markdown / JSON)
```

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – Web UI
* **FAISS** – Vector database
* **LangChain-style modular RAG**
* **Ollama (LLaMA 3 8B)** – Local LLM (optional)
* **OpenAI API** – Optional cloud generation
* **dotenv** – Environment management

---

## 📂 Project Structure

```
.
├── streamlit_app.py          # Main Streamlit app
├── roadmap_rag/
│   ├── ingest.py             # Data ingestion
│   ├── chunk.py              # Text chunking
│   ├── embed.py              # Embedding logic
│   ├── retrieve.py           # Top-k retrieval
│   ├── generate.py           # Roadmap generation
│   └── store.py              # FAISS local store
├── data/seed/                # Local seed documents
├── index/                    # FAISS index
├── .env                      # Environment variables
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ (Optional) Set environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

> If no API key is provided, the app works in **offline mode** using local embeddings & templated generation.

### 3️⃣ Run the app

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Usage

1. Add seed URLs or local documents
2. Click **Rebuild Index**
3. Choose chunk size, overlap, and `k`
4. Ask questions like:

```
Create a backend roadmap for an intermediate developer
```

5. Export the generated roadmap in Markdown or JSON

---

## 📈 Future Enhancements

* 🧩 Skill-gap quizzes for automatic level detection
* 📱 Mobile app with offline sync
* 🧠 Learning progress tracking
* 🧪 Roadmap evaluation metrics

---


## ⭐ Acknowledgments

* FAISS by Meta
* Streamlit
* LangChain concepts
* Ollama for local LLMs

---

> 💡 *Roadmaps should adapt to you — not the other way around.*
