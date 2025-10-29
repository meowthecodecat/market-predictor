# backend/scripts/predict_xxx.py
from pathlib import Path
import pandas as pd
from datetime import datetime

def save_run_preds(pred_rows):
    # pred_rows: liste de dicts {"p_hat": float, "y_true": int}
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path(__file__).resolve().parents[1] / "data" / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows)[["p_hat","y_true"]].to_csv(out_dir / f"preds__{ts}.csv", index=False)
