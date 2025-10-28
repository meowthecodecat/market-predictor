# FILENAME: scripts/evaluate_backtest.py
# -*- coding: utf-8 -*-
"""
Backtest simple avec sizing et stops ATR.
- Position = weight_t * sign(ret_final) (weight borné [0,1])
- SL = 1.5*ATR, TP = 3*ATR, time-stop = 5 jours
- Exporte metrics/jour dans reports/daily_metrics.csv
Entrées attendues: data/*_historical_prices.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def _load(sym: str) -> pd.DataFrame:
    p = DATA / f"{sym}_historical_prices.csv"
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

def _atr14(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    pc = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=1).mean().fillna(method="bfill").fillna(0.0)

def backtest_symbol(sym: str) -> pd.DataFrame:
    df = _load(sym)
    df["ATR_14"] = _atr14(df)
    close = df["Close"].values
    ret1 = pd.Series(close).pct_change().shift(-1).fillna(0.0).values

    # signal proxy: direction = signe de ret_5 récent, sizing via ATR
    ret5 = pd.Series(close).pct_change(5).fillna(0.0).values
    p_up = 1/(1+np.exp(-20*ret5))
    conf = np.maximum(0.0, (p_up - 0.5) * 2.0)  # 0..1
    VOL_TARGET_DAILY = 0.15/np.sqrt(252)
    weight = np.minimum(1.0, VOL_TARGET_DAILY / (df["ATR_14"].replace(0.0, np.nan))).fillna(0.0).values
    pos = np.sign(ret5) * conf * weight  # [-1,1] borné

    # SL/TP ATR, time-stop=5j (approx en ret quotidien)
    sl = 1.5 * df["ATR_14"].values / close
    tp = 3.0 * df["ATR_14"].values / close
    max_hold = 5

    pnl = np.zeros(len(df))
    holding = 0
    entry_idx = None
    direction = 0

    for t in range(len(df)-1):
        if holding == 0:
            direction = np.sign(pos[t])  # -1, 0, 1
            if direction != 0:
                holding = 1
                entry_idx = t
                entry_price = close[t]
        else:
            # mouvement du jour t->t+1
            move = (close[t+1] - close[t]) / close[t]
            pnl[t+1] += direction * weight[t] * move
            # SL/TP check
            if direction > 0:
                if (close[t+1] - entry_price)/entry_price <= -sl[entry_idx]: holding = 0
                elif (close[t+1] - entry_price)/entry_price >=  tp[entry_idx]: holding = 0
            else:
                if (entry_price - close[t+1])/entry_price <= -sl[entry_idx]: holding = 0
                elif (entry_price - close[t+1])/entry_price >=  tp[entry_idx]: holding = 0
            # time-stop
            if holding and (t - entry_idx + 1) >= max_hold: holding = 0

    eq = (1.0 + pd.Series(pnl)).cumprod()
    ret = pd.Series(pnl).replace([np.inf,-np.inf], 0.0).fillna(0.0)
    sharpe = (ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252) if len(ret)>2 else 0.0
    mdd = ((eq / eq.cummax()) - 1.0).min() if len(eq) else 0.0

    daily = pd.DataFrame({
        "Date": df["Date"],
        "symbol": sym,
        "ret": ret.values,
        "weight": weight,
        "pos": pos,
        "ATR_14": df["ATR_14"].values
    })
    daily["equity"] = (1.0 + daily["ret"]).cumprod()
    daily["sharpe_rolling"] = daily["ret"].rolling(60).mean() / (daily["ret"].rolling(60).std()+1e-12) * np.sqrt(252)

    print(f"{sym}  Sharpe={sharpe:.2f}  MaxDD={mdd:.2%}  CAGR={float(eq.iloc[-1]-1):.2%}")
    return daily

def main():
    tickers = ["AAPL","NVDA","AMD","TSLA","AMZN","GOOGL","MSFT","KO"]
    frames = [backtest_symbol(s) for s in tickers]
    daily = pd.concat(frames, ignore_index=True)
    out = REPORTS / "daily_metrics.csv"
    daily.to_csv(out, index=False)
    print(f"Saved daily metrics: {out}")

if __name__ == "__main__":
    main()
