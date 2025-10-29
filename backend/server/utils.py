from pathlib import Path
import pandas as pd

def latest_pred_files_per_day(pred_dir: Path):
    files = list(pred_dir.glob("preds__*.csv"))
    if not files:
        return []
    df = pd.DataFrame({"path": files})
    df["ts"] = pd.to_datetime(df["path"].map(lambda p: p.stem.split("__")[-1]), errors="coerce")
    df["day"] = df["ts"].dt.date
    return df.sort_values("ts").groupby("day", as_index=False).tail(1)["path"].tolist()

def calibration_bins(df, edges=(0.5,0.6,0.7,0.8,0.9,1.0)):
    cuts = pd.cut(df["p_hat"].clip(0,1), bins=[0.0,*edges], include_lowest=True, right=True)
    g = df.groupby(cuts, observed=True)
    out = g.agg(n=("y_true","size"), acc=("y_true","mean")).reset_index(names="bin")
    out["bin_left"]  = out["bin"].map(lambda x: float(x.left))
    out["bin_right"] = out["bin"].map(lambda x: float(x.right))
    return out[["bin_left","bin_right","n","acc"]]
