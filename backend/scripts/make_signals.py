# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IN_CSV = DATA / "run_summary.csv"
OUT_CSV = DATA / "signals.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up-th", type=float, default=0.20, help="d_pct ≥ th -> BUY")
    ap.add_argument("--down-th", type=float, default=-0.20, help="d_pct ≤ th -> SELL")
    args = ap.parse_args()

    if not IN_CSV.exists():
        print(f"introuvable: {IN_CSV}")
        return

    df = pd.read_csv(IN_CSV)
    # on ne garde que les lignes valides
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        print("Aucune ligne OK encore disponible.")
        return

    df["d_pct"] = pd.to_numeric(df["d_pct"], errors="coerce")
    def decide(x):
        if x >= args.up_th: return "BUY"
        if x <= args.down_th: return "SELL"
        return "HOLD"

    df["signal"] = df["d_pct"].map(decide)
    out = df[["ts_utc","symbol","last_close","pred_close","d_pct","signal"]].copy()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Signaux écrits: {OUT_CSV}")

if __name__ == "__main__":
    main()
