# backend/server/utils.py
from pathlib import Path
from typing import List, Dict
import pandas as pd
import re
from datetime import datetime, timezone

def read_summary_csv(csv_path: Path) -> List[Dict]:
    df = pd.read_csv(csv_path)
    cols = ["ts_utc", "symbol", "last_close", "pred_close", "d_pct", "status"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["ts_utc"] = df["ts_utc"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    for c in ["last_close", "pred_close", "d_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["status"] = df["status"].astype(str)
    records = []
    for _, r in df[cols].iterrows():
        records.append(
            {
                "ts_utc": r["ts_utc"],
                "symbol": r["symbol"],
                "last_close": float(r["last_close"]) if pd.notna(r["last_close"]) else 0.0,
                "pred_close": float(r["pred_close"]) if pd.notna(r["pred_close"]) else 0.0,
                "d_pct": float(r["d_pct"]) if pd.notna(r["d_pct"]) else 0.0,
                "status": r["status"] if isinstance(r["status"], str) else "NA",
            }
        )
    return records

def detect_models(models_dir: Path) -> List[Dict]:
    models_dir.mkdir(parents=True, exist_ok=True)
    files = list(models_dir.glob("*"))
    by_symbol: Dict[str, Dict] = {}
    pat = re.compile(r"^([A-Z]{1,10})", re.IGNORECASE)
    for p in files:
        name = p.name
        m = pat.match(name)
        if not m:
            continue
        sym = m.group(1).upper()
        rec = by_symbol.setdefault(sym, {"symbol": sym, "lstm": False, "scaler": False, "updated_at": None})
        if name.lower().endswith(".keras"):
            rec["lstm"] = True
        if name.lower().endswith(".pkl"):
            rec["scaler"] = True
        ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        # garder le plus récent
        if rec["updated_at"] is None or ts > rec["updated_at"]:
            rec["updated_at"] = ts
    return sorted(by_symbol.values(), key=lambda x: x["symbol"])
