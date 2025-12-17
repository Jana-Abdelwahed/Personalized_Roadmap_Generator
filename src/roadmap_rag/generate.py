import os
from typing import List, Tuple
from textwrap import dedent

def generate_answer(question: str, context_chunks: List[Tuple[str, str]], model: str = "gpt-4o-mini") -> str:
    """Use OpenAI if available; otherwise, return a templated answer that cites sources."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    system = dedent("""
    You are a helpful AI engineering tutor. Answer using only the provided CONTEXT.
    If the answer is not in the context, say you don't know.
    Cite sources as [source:N] where N matches the provided source indices.
    """).strip()

    context_text = "\n\n".join([f"[source:{i}] {c[:1200]}" for i, (_, c) in enumerate(context_chunks)])
    prompt = dedent(f"""
    CONTEXT:
    {context_text}

    QUESTION: {question}
    """).strip()

    if not api_key:
        bullets = "\n".join([f"- Uses [source:{i}]" for i in range(len(context_chunks))])
        return dedent(f"""
        (Offline demo mode - no OPENAI_API_KEY)
        Here's a synthesized answer based on retrieved context:
        - Key points are grounded in the provided sources.
        {bullets}
        """).strip()

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Generation failed: {e}"
