from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.loaders import load_file, chunk_records
from src.rag import QualityRAG, extractive_answer, ollama_answer, build_query
from src.analytics import records_dataframe, pareto

st.set_page_config(page_title="NCR / CAPA / Scrap Root-Cause RAG", page_icon="🔎", layout="wide")
st.title("NCR / CAPA / Scrap Root-Cause RAG")
st.caption("Local-first quality intelligence demo • evidence retrieval • root-cause support • no paid API required")

if "records" not in st.session_state:
    st.session_state.records = []
if "rag" not in st.session_state:
    st.session_state.rag = QualityRAG()

with st.sidebar:
    st.header("1. Load quality data")
    uploaded = st.file_uploader(
        "Upload NCR / CAPA / scrap files",
        type=["csv", "xlsx", "xlsm", "pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded files", type="primary", use_container_width=True):
        records = []
        for f in uploaded or []:
            try:
                records.extend(load_file(f.name, f.getvalue()))
            except Exception as e:
                st.error(f"{f.name}: {e}")
        records = chunk_records(records)
        st.session_state.records = records
        st.session_state.rag.fit(records)
        st.success(f"Indexed {len(records)} searchable records/chunks.")

    if st.button("Load sample data", use_container_width=True):
        sample = Path("data/sample/quality_events.csv")
        records = load_file(sample.name, sample.read_bytes())
        records = chunk_records(records)
        st.session_state.records = records
        st.session_state.rag.fit(records)
        st.success(f"Indexed {len(records)} sample records.")

    st.divider()
    st.header("2. Answer mode")
    mode = st.radio("Mode", ["Free extractive RAG", "Ollama local LLM"])
    model = st.text_input("Ollama model", "llama3.2:3b", disabled=mode != "Ollama local LLM")
    top_k = st.slider("Evidence records", 3, 10, 5)

records = st.session_state.records
if not records:
    st.info("Upload your quality records or click **Load sample data** to start.")
else:
    df = records_dataframe(records)
    c1, c2, c3 = st.columns(3)
    c1.metric("Indexed records", len(records))
    c2.metric("Source files", df["source"].nunique() if "source" in df else 0)
    c3.metric("Mode", "Local / private")

    tabs = st.tabs(["Ask Root-Cause RAG", "Similar Cases", "Quality Analytics", "Data Preview", "How it works"])

    with tabs[0]:
        st.subheader("Ask about NCR, CAPA, scrap, defects, containment, or recurrence")
        question = st.text_area(
            "Question",
            "What recurring root causes are associated with dimensional defects, and what corrective actions worked?",
            height=100,
        )
        part = st.text_input("Optional part / product")
        process = st.text_input("Optional process / machine")
        if st.button("Analyze", type="primary"):
            query = build_query(question, part, process)
            hits = st.session_state.rag.search(query, top_k=top_k)
            if mode == "Ollama local LLM":
                try:
                    answer = ollama_answer(question, hits, model=model)
                except Exception as e:
                    st.warning(f"Ollama unavailable ({e}). Falling back to extractive RAG.")
                    answer = extractive_answer(question, hits)
            else:
                answer = extractive_answer(question, hits)
            st.markdown("### Answer")
            st.write(answer)
            st.markdown("### Evidence")
            for i, h in enumerate(hits, 1):
                with st.expander(f"{i}. {h['source']} · {h['record_id']} · relevance {h['score']:.3f}"):
                    st.text(h["text"])

    with tabs[1]:
        st.subheader("Find similar historical quality events")
        case = st.text_area("Describe the new issue", "Bore diameter oversize after tool change; repeated on night shift.")
        if st.button("Find similar cases"):
            hits = st.session_state.rag.search(case, top_k=top_k)
            for h in hits:
                st.markdown(f"**{h['source']} — {h['record_id']}** · relevance `{h['score']:.3f}`")
                st.write(h["text"][:1200])
                st.divider()

    with tabs[2]:
        st.subheader("Quality analytics")
        p = pareto(df)
        if p.empty:
            st.info("For Pareto analytics, include a column such as Defect / Defect_Type / Failure_Mode in CSV/XLSX data.")
        else:
            fig = px.bar(p, x=p.columns[0], y="impact", title="Defect Pareto")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(p, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[4]:
        st.markdown("""
**Pipeline:** files → parsing → record chunking → local TF-IDF vectorization → cosine-similarity retrieval → evidence-backed answer.

**Why this demo is safe for a free prototype:** no paid model API is required. The default answer is extractive and always exposes the retrieved source evidence. For generative synthesis, run Ollama locally and switch the sidebar mode.

**Production upgrades:** add SSO/RBAC, immutable source IDs, approval workflow, audit logs, controlled taxonomy, validated embeddings/vector DB, document versioning, human CAPA approval, and your QMS/MES/ERP connectors.
""")
