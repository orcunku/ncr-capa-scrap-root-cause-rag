from __future__ import annotations
from pathlib import Path
import io
import pandas as pd
from pypdf import PdfReader
from docx import Document

TEXT_EXTENSIONS = {".txt", ".md", ".log"}

def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())

def dataframe_to_records(df: pd.DataFrame, source: str) -> list[dict]:
    records = []
    for i, row in df.fillna("").iterrows():
        fields = [f"{c}: {_clean(row[c])}" for c in df.columns if _clean(row[c])]
        if fields:
            records.append({
                "source": source,
                "record_id": str(row.get("NCR_ID", row.get("CAPA_ID", row.get("SCRAP_ID", i + 1)))),
                "text": "\n".join(fields),
                "metadata": {str(c): _clean(row[c]) for c in df.columns},
            })
    return records

def load_file(name: str, raw: bytes) -> list[dict]:
    ext = Path(name).suffix.lower()
    bio = io.BytesIO(raw)
    if ext == ".csv":
        return dataframe_to_records(pd.read_csv(bio), name)
    if ext in {".xlsx", ".xlsm"}:
        xls = pd.ExcelFile(bio)
        out = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            out.extend(dataframe_to_records(df, f"{name}::{sheet}"))
        return out
    if ext == ".pdf":
        reader = PdfReader(bio)
        out = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                out.append({"source": name, "record_id": f"page-{i+1}", "text": text, "metadata": {"page": i+1}})
        return out
    if ext == ".docx":
        doc = Document(bio)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"source": name, "record_id": "document", "text": text, "metadata": {}}] if text else []
    if ext in TEXT_EXTENSIONS:
        text = raw.decode("utf-8", errors="ignore")
        return [{"source": name, "record_id": "document", "text": text, "metadata": {}}] if text.strip() else []
    raise ValueError(f"Unsupported file type: {ext}")

def chunk_records(records: list[dict], max_chars: int = 1800, overlap: int = 250) -> list[dict]:
    chunks = []
    for rec in records:
        text = rec["text"]
        if len(text) <= max_chars:
            chunks.append(rec)
            continue
        start, n = 0, 1
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunk = dict(rec)
            chunk["text"] = text[start:end]
            chunk["record_id"] = f'{rec["record_id"]}-chunk-{n}'
            chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - overlap)
            n += 1
    return chunks
