# FILENAME: scripts/threshold_sweep.py
# -*- coding: utf-8 -*-
# Déterministe (pas d'aléa)
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def backtest_long_only(returns: np.ndarray, signals: np.ndarray, fees_bps: float):
    s = signals.astype(int)
    turns = np.diff(s, prepend=0) != 0
    daily = returns * s - fees_bps * turns
    eq = (1.0 + pd.Series(daily)).cumprod()
    perf = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
    ret = pd.Series(daily).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    sharpe = float((ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252)) if len(ret) > 2 else 0.0
    peak = eq.cummax()
    mdd = float(((eq / peak) - 1.0).min()) if len(eq) else 0.0
    return perf, sharpe, -mdd

def make_regimes(close: pd.Series):
    ret1 = close.pct_change().fillna(0.0)
    vol20 = ret1.rolling(20, min_periods=1).std().fillna(0.0)
    vol_reg = pd.qcut(vol20, 3, labels=[0,1,2], duplicates="drop").astype("Int8")
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    trend_reg = np.sign(ema12 - ema26).astype("Int8")
    return vol_reg, trend_reg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fees", type=float, default=0.0002)
    ap.add_argument("--symbol", type=str, default="AAPL")
    args = ap.parse_args()

    p = DATA / f"{args.symbol}_historical_prices.csv"
    if not p.exists():
        print("CSV prix introuvable.")
        return
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

    close = pd.to_numeric(df["Close"], errors="coerce")
    ret1 = close.pct_change().shift(-1).fillna(0.0)
    vol_reg, trend_reg = make_regimes(close)

    # proxy p_up déterministe
    k = 20.0
    ret5 = close.pct_change(5).fillna(0.0)
    p_up = (1.0 / (1.0 + np.exp(-k * ret5))).to_numpy()

    rows = []
    for v in [0,1,2]:
        for t in [-1,0,1]:
            mask = (vol_reg.to_numpy() == v) & (trend_reg.to_numpy() == t)
            if mask.sum() < 30:
                continue
            ret_sub = ret1.to_numpy()[mask]
            p_sub = p_up[mask]
            for thr in np.round(np.arange(0.45, 0.76, 0.01), 3):
                sig = (p_sub >= thr).astype(int)
                perf, sharpe, mdd = backtest_long_only(ret_sub, sig, args.fees)
                rows.append([v, t, thr, perf, sharpe, mdd, int(sig.sum()), int(mask.sum())])

    out = pd.DataFrame(rows, columns=["vol_reg","trend_reg","thr","cum_perf","sharpe","max_dd","trades","n"])
    out = out.sort_values(["sharpe","cum_perf"], ascending=False)
    out.to_csv(DATA / "threshold_sweep_results_grid.csv", index=False)

    print("TOP 10 global:")
    print(out.head(10).to_string(index=False))

    print("\nSeuils recommandés par cellule (meilleur Sharpe):")
    best = out.sort_values(["vol_reg","trend_reg","sharpe","cum_perf"], ascending=[True,True,False,False]) \
              .groupby(["vol_reg","trend_reg"]).head(1)
    print(best[["vol_reg","trend_reg","thr","sharpe","cum_perf","max_dd","trades","n"]].to_string(index=False))

if __name__ == "__main__":
    main()
