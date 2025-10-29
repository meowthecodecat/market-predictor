# FILENAME: scripts/evaluate_predictions.py
# -*- coding: utf-8 -*-
"""
Met à jour data/model_errors.csv en comparant les prédictions passées à la réalité.
Méthode:
- Pour chaque ligne de data/run_summary.csv (ts_utc, symbol, last_close, pred_close):
  1) ts_date = date(ts_utc)
  2) Dans data/{symbol}_historical_prices.csv, on prend D = dernière Date <= ts_date
  3) Si D+1 existe, actual_close = Close[D+1], sinon on skip
  4) pred_ret = pred_close/last_close - 1 ; actual_ret = actual_close/close[D] - 1
  5) err_1 = actual_ret - pred_ret ; hit = 1 si sign(pred_ret)==sign(actual_ret)
- Puis on calcule (par symbol, tri par Date):
  err_ma5, err_ma20, bias20 (=moyenne signée 20j), hit20 (=moyenne des hits 20j)
- Sauvegarde dans data/model_errors.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUN_SUMMARY = DATA / "run_summary.csv"
OUT = DATA / "model_errors.csv"

def _load_prices(sym: str) -> pd.DataFrame:
    p = DATA / f"{sym}_historical_prices.csv"
    df = pd.read_csv(p)
    df.columns = [c.capitalize() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

def main():
    if not RUN_SUMMARY.exists():
        return 0

    rs = pd.read_csv(RUN_SUMMARY)
    # Guard minimal
    need_cols = {"ts_utc","symbol","last_close","pred_close"}
    if not need_cols.issubset(set(rs.columns)):
        return 0

    rs = rs.copy()
    rs["ts_utc"] = pd.to_datetime(rs["ts_utc"], errors="coerce")
    rs = rs.dropna(subset=["ts_utc","symbol","last_close","pred_close"])
    rs["pred_ret"] = rs["pred_close"].astype(float) / rs["last_close"].astype(float) - 1.0
    rs["ts_date"] = rs["ts_utc"].dt.normalize()

    rows = []
    for sym, grp in rs.groupby("symbol"):
        prices = _load_prices(sym)
        close = prices["Close"].to_numpy()
        dates = prices["Date"]
        for _, r in grp.iterrows():
            ts_date = r["ts_date"]
            # D = dernière date <= ts_date
            idx = dates.searchsorted(ts_date, side="right") - 1
            if idx < 0 or idx >= len(dates) - 1:
                continue
            d = dates.iloc[idx]
            d1 = dates.iloc[idx + 1]
            last_close_run = float(close[idx])
            actual_close = float(close[idx + 1])
            actual_ret = actual_close / last_close_run - 1.0
            pred_ret = float(r["pred_ret"])
            err_1 = actual_ret - pred_ret
            hit = 1 if np.sign(actual_ret) == np.sign(pred_ret) and pred_ret != 0 else 0
            rows.append({
                "Date": d, "Symbol": sym,
                "pred_ret": pred_ret, "actual_ret": actual_ret,
                "err_1": err_1, "abs_err_1": abs(err_1), "hit": hit
            })

    if not rows:
        return 0

    df = pd.DataFrame(rows).sort_values(["Symbol","Date"]).reset_index(drop=True)

    # Merge avec historique existant
    if OUT.exists():
        old = pd.read_csv(OUT, parse_dates=["Date"])
        all_df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["Date","Symbol"], keep="last")
        df = all_df.sort_values(["Symbol","Date"]).reset_index(drop=True)

    # Rolling metrics par symbol
    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").reset_index(drop=True)
        g["err_ma5"] = g["err_1"].rolling(5, min_periods=1).mean()
        g["err_ma20"] = g["err_1"].rolling(20, min_periods=1).mean()
        g["bias20"] = g["err_1"].rolling(20, min_periods=1).mean()
        g["hit20"] = g["hit"].rolling(20, min_periods=1).mean()
        return g
    df = df.groupby("Symbol", group_keys=False).apply(_roll)

    df.to_csv(OUT, index=False)
    print(f"Updated: {OUT} ({len(df)} rows)")
    return 0

if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
