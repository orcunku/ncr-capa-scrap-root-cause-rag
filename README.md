# NCR / CAPA / Scrap Root-Cause RAG

A free, local-first Streamlit demo for searching NCR, CAPA, scrap and manufacturing quality history, finding similar cases, and grounding root-cause analysis in evidence.

👉 [Open the Live Streamlit Application](https://ncr-capa-scrap-root-cause-rag-njjotdgvsfylcd5cosiigh.streamlit.app/)

A free, local-first Streamlit demo for searching NCR, CAPA, scrap and manufacturing quality history, finding similar cases, and grounding root-cause analysis in evidence.

### 🎯 What It Does

- Searches historical NCR, CAPA and scrap records
- Identifies similar quality incidents
- Retrieves evidence for root-cause investigations
- Finds previous corrective and preventive actions
- Provides defect Pareto analysis
- Supports CSV, Excel, PDF, DOCX and text records
- Runs without a paid AI API

> This is a decision-support demo, not an autonomous CAPA approval system. Validate source records and keep human quality approval in the loop.

## Architecture

```text
NCR / CAPA / Scrap files
        |
        v
  File loaders
 CSV/XLSX/PDF/DOCX/TXT
        |
        v
 Record normalization + chunking
        |
        v
 TF-IDF local vector index
        |
        +--------------------+
        |                    |
        v                    v
 Similar-case search    Root-cause question
        |                    |
        +---------> Retrieval top-k
                             |
                  +----------+-----------+
                  |                      |
                  v                      v
           Extractive answer      Optional Ollama
           (default/free)         local synthesis
                  |                      |
                  +----------+-----------+
                             v
                   Streamlit evidence UI
```

## Quick start

Python 3.10+ is recommended.

```bash
git clone <YOUR-REPO-URL>
cd ncr-capa-scrap-rag

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then click **Load sample data**.

## Optional: local LLM with Ollama

Install Ollama separately, then pull a small model:

```bash
ollama pull llama3.2:3b
ollama serve
```

Open the app and choose **Ollama local LLM** in the sidebar. The app calls the local Ollama HTTP API at `http://localhost:11434`.

If Ollama is unavailable, the app automatically falls back to extractive RAG.

## Your data schema

There is no mandatory schema for retrieval. For the best demo, structured NCR/CAPA/scrap files can include columns like:

```text
NCR_ID, Date, Part, Process, Machine, Defect, Problem,
Root_Cause, Containment, Corrective_Action, Preventive_Action,
Scrap_Qty, Status
```

The Pareto view recognizes common defect/failure-mode and quantity column names.

## Free deployment

### Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create an app from the GitHub repository.
4. Set the entry point to `streamlit_app.py`.
5. Deploy.

The default extractive mode works without API keys. A local Ollama server is normally not reachable from Streamlit Community Cloud, so use extractive mode for the fully free hosted demo.

### Docker

```bash
docker build -t quality-rag .
docker run --rm -p 8501:8501 quality-rag
```

## GitHub commands

```bash
git init
git add .
git commit -m "Initial NCR CAPA Scrap Root-Cause RAG demo"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ncr-capa-scrap-rag.git
git push -u origin main
```

## Suggested production roadmap

1. Replace session-only indexing with persistent storage.
2. Add authenticated users, RBAC and site/business-unit filtering.
3. Add controlled defect/root-cause taxonomies and data validation.
4. Add source-document versioning, audit trails and approval workflows.
5. Add QMS/MES/ERP ingestion connectors.
6. Add embedding-based semantic retrieval and reranking after validation.
7. Add structured 5-Why / Ishikawa assistance and recurrence detection.
8. Add CAPA effectiveness checks and due-date tracking.
9. Add evaluation datasets: retrieval precision, citation correctness, hallucination rate.
10. Complete cybersecurity, privacy, validation and regulatory review before production use.

## Tests

```bash
pip install pytest
pytest -q
```

## License

MIT
