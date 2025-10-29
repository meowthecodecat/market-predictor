# backend/scripts/metrics.py
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict

def load_runs(csv_path: Path) -> List[Dict]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def directional_accuracy(rows: List[Dict], symbol: str | None = None) -> List[Dict]:
    # attend colonnes: ts_utc, symbol, last_close, pred_close, actual_close
    out = {}
    for row in rows:
        if symbol and row["symbol"] != symbol:
            continue
        try:
            ts = datetime.fromisoformat(row["ts_utc"])
            last_c = float(row["last_close"])
            pred_c = float(row["pred_close"])
            act_c  = float(row["actual_close"])
        except Exception:
            continue
        pred_up = pred_c >= last_c
        act_up  = act_c  >= last_c
        ok = 1 if (pred_up == act_up) else 0
        key = ts.date().isoformat()
        out.setdefault(key, []).append(ok)
    series = []
    for day, oks in sorted(out.items()):
        acc = sum(oks) / len(oks)
        series.append({"date": day, "accuracy": round(acc, 4)})
    return series
