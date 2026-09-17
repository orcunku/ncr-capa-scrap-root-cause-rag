from __future__ import annotations
import pandas as pd

def records_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"source": r["source"], "record_id": r["record_id"], **r.get("metadata", {})}
        rows.append(row)
    return pd.DataFrame(rows)

def find_col(df: pd.DataFrame, candidates: list[str]):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None

def pareto(df: pd.DataFrame):
    defect = find_col(df, ["defect", "defect_type", "failure_mode", "issue", "nonconformance"])
    qty = find_col(df, ["scrap_qty", "quantity", "qty", "scrap_quantity"])
    if defect is None:
        return pd.DataFrame()
    if qty:
        p = df.groupby(defect, dropna=False)[qty].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
    else:
        p = df.groupby(defect, dropna=False).size()
    p = p.sort_values(ascending=False).reset_index(name="impact")
    p["cumulative_pct"] = 100 * p["impact"].cumsum() / max(p["impact"].sum(), 1)
    return p
