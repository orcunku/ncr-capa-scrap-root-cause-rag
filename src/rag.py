from __future__ import annotations
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class QualityRAG:
    """Zero-cost retrieval layer: local TF-IDF index + optional local Ollama generation."""
    def __init__(self):
        self.records = []
        self.vectorizer = None
        self.matrix = None

    def fit(self, records: list[dict]):
        self.records = records
        if not records:
            self.vectorizer = self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, max_features=40000, sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform([r["text"] for r in records])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.matrix is None or not query.strip():
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        ids = np.argsort(scores)[::-1][:top_k]
        return [{**self.records[i], "score": float(scores[i])} for i in ids if scores[i] > 0]

def extractive_answer(question: str, hits: list[dict]) -> str:
    if not hits:
        return "No relevant evidence was found in the indexed quality records."
    lines = []
    for h in hits[:4]:
        text = re.sub(r"\s+", " ", h["text"]).strip()
        excerpt = text[:650] + ("…" if len(text) > 650 else "")
        lines.append(f'- [{h["source"]} / {h["record_id"]}] {excerpt}')
    return (
        "Evidence-backed findings from the most relevant NCR/CAPA/Scrap records:\n\n"
        + "\n\n".join(lines)
        + "\n\nUse these records as the audit trail; verify the original source before approving a CAPA or disposition."
    )

def ollama_answer(question: str, hits: list[dict], model: str = "llama3.2:3b", base_url: str = "http://localhost:11434") -> str:
    import requests
    context = "\n\n".join(
        f"SOURCE {i+1}: {h['source']} | {h['record_id']}\n{h['text']}" for i, h in enumerate(hits)
    )
    prompt = f"""You are a manufacturing quality root-cause assistant.
Answer ONLY from the supplied evidence. Do not invent facts.
Separate: observed issue, likely root cause, containment, corrective action, preventive action, and evidence gaps.
Cite sources inline as [SOURCE 1], [SOURCE 2], etc.
If evidence is insufficient, say so.

QUESTION:
{question}

EVIDENCE:
{context}
"""
    r = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]

def build_query(issue: str, part: str = "", process: str = "", defect: str = "") -> str:
    return " ".join(x for x in [issue, part, process, defect] if x).strip()
