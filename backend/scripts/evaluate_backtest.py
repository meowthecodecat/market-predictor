# FILENAME: scripts/evaluate_backtest.py
# -*- coding: utf-8 -*-
"""
Évaluation prix et backtest simple avec découpes par régimes.

Usage:
  python scripts/evaluate_backtest.py
  # options:
  #   --data-dir data --fees 0.0002

Lit *_historical_prices.csv dans data/.
Calcule:
  - Buy&Hold cumulé, Sharpe, MaxDD
  - Stats par régime de tendance (EMA12 vs EMA26)
  - Stats par régime de volatilité (σ20 sur ret1, terciles)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def _load_one(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"Colonnes manquantes: {p}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return df

def _sharpe(ret: pd.Series) -> float:
    ret = ret.replace([np.inf,-np.inf], 0.0).fillna(0.0)
    s = ret.std()
    return float((ret.mean() / (s + 1e-12)) * np.sqrt(252)) if len(ret) > 2 else 0.0

def _maxdd(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq/peak - 1.0).min()
    return float(-dd)

def _bh_stats(close: pd.Series) -> dict:
    ret1 = close.pct_change().fillna(0.0)
    eq = (1.0 + ret1).cumprod()
    return {
        "n_days": int((ret1.shape[0])),
        "bh_cum": float(eq.iloc[-1] - 1.0),
        "sharpe": _sharpe(ret1),
        "maxdd": _maxdd(eq),
        "avg_ret_1d": float(ret1.mean()),
        "vol_1d": float(ret1.std()),
    }

def _regimes(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema12"] = d["Close"].ewm(span=12, adjust=False).mean()
    d["ema26"] = d["Close"].ewm(span=26, adjust=False).mean()
    d["trend_regime"] = np.sign(d["ema12"] - d["ema26"]).astype("Int8")  # -1,0,1
    ret1 = d["Close"].pct_change().fillna(0.0)
    vol20 = ret1.rolling(20, min_periods=1).std().fillna(0.0)
    d["vol_regime"] = pd.qcut(vol20, 3, labels=[0,1,2], duplicates="drop").astype("Int8")  # 0=bas,2=haut
    return d

def _stats_by_group(df: pd.DataFrame, key: str) -> pd.DataFrame:
    out = []
    for k, g in df.groupby(key):
        st = _bh_stats(g["Close"])
        st["group"] = int(k)
        out.append(st)
    return pd.DataFrame(out).sort_values("group")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--fees", type=float, default=0.0000, help="frais fictifs non utilisés ici, placeholder")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*_historical_prices.csv"))
    if not files:
        print("Aucun CSV *_historical_prices.csv trouvé.")
        return

    print("symbol,n_days,bh_cum,sharpe,maxdd,avg_ret_1d,vol_1d")
    for f in files:
        sym = f.name.split("_")[0]
        try:
            df = _load_one(f)
            base = _bh_stats(df["Close"])
            print(f"{sym},{base['n_days']},{base['bh_cum']:.6f},{base['sharpe']:.3f},{base['maxdd']:.3f},{base['avg_ret_1d']:.6f},{base['vol_1d']:.6f}")

            # Régimes
            d = _regimes(df)
            by_trend = _stats_by_group(d, "trend_regime")
            by_vol = _stats_by_group(d, "vol_regime")

            out_tr = data_dir / f"{sym}_stats_by_trend.csv"
            out_vl = data_dir / f"{sym}_stats_by_vol.csv"
            by_trend.to_csv(out_tr, index=False)
            by_vol.to_csv(out_vl, index=False)
        except Exception as e:
            print(f"{sym},0,,,,,  # FAIL: {e}")

if __name__ == "__main__":
    main()
