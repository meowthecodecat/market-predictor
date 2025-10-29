# backend/scripts/predict_next_close.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import make_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

def _load_prices(symbol: str) -> pd.DataFrame:
    """Combine l'historique daily complet avec le daily_live du jour si présent."""
    hist_p = DATA / f"{symbol}_historical_prices.csv"
    live_p = DATA / f"{symbol}_daily_live.csv"
    if not hist_p.exists():
        raise FileNotFoundError(f"Manque historique: {hist_p}")

    hist = pd.read_csv(hist_p)
    hist = hist.rename(columns={c: c.capitalize() for c in hist.columns})
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce").dt.normalize()
    hist = hist.dropna(subset=["Date","Close"]).sort_values("Date")

    if live_p.exists():
        live = pd.read_csv(live_p)
        live = live.rename(columns={c: c.capitalize() for c in live.columns})
        live["Date"] = pd.to_datetime(live["Date"], errors="coerce").dt.normalize()
        live = live.dropna(subset=["Date","Close"]).sort_values("Date")
        if not live.empty:
            cutoff = live["Date"].min()
            hist = hist[hist["Date"] < cutoff]
            combined = pd.concat([hist[["Date","Open","High","Low","Close","Volume"]],
                                  live[["Date","Open","High","Low","Close","Volume"]]],
                                 ignore_index=True).sort_values("Date")
            return combined.reset_index(drop=True)

    return hist.reset_index(drop=True)

def _load_errors(sym: str) -> pd.DataFrame | None:
    p = DATA / "model_errors.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    df = df[df["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
    df = df.drop(columns=["Symbol"], errors="ignore")
    return df

def _to_seq_last(X2: np.ndarray, T: int) -> np.ndarray:
    if len(X2) < T:
        raise ValueError(f"Pas assez d'observations pour T={T} (n={len(X2)})")
    return X2[-T:, :][None, :, :]

def _residual_sigma(model, X_all_s: np.ndarray, y_all: np.ndarray, T: int) -> float:
    xs, ys = [], []
    for i in range(T, len(X_all_s)):
        xs.append(X_all_s[i - T : i, :])
        ys.append(y_all[i])
    if not xs:
        return 0.0
    Xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    yhat = model.predict(Xs, verbose=0).reshape(-1)
    resid = ys - yhat
    return float(np.nanstd(resid))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--time-step", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.6)  # conservé pour compat
    args = ap.parse_args()

    sym = args.symbol.upper()
    model_path = MODELS / f"nextclose_{sym}.keras"
    scaler_path = MODELS / f"nextclose_{sym}_scaler.pkl"
    feats_path = MODELS / f"nextclose_{sym}_features.json"
    if not (model_path.exists() and scaler_path.exists() and feats_path.exists()):
        print(f"{sym} -> modèles/scaler/features manquants")
        return 1

    with open(feats_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    use_cols = meta.get("features", [])
    T = int(meta.get("time_step", args.time_step))

    raw = _load_prices(sym)
    err = _load_errors(sym)

    # exog facultatifs
    def _try(symbol: str) -> pd.DataFrame | None:
        p = DATA / f"{symbol}_historical_prices.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.capitalize() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
            return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
        return None
    exog = {k: _try(k) for k in ["SPY","XLK","DXY","US10Y"] if _try(k) is not None}

    feat_df, _ = make_features(raw, market_df=None, earnings_dates=None, exog=exog, errors_df=err)
    last_close = float(raw["Close"].iloc[-1])

    use_cols = [c for c in use_cols if c in feat_df.columns]
    if not use_cols:
        print(f"{sym} -> features absentes dans le dataset.")
        return 1

    # convertit en numpy avant scaler pour éviter le warning scikit-learn
    X_all = feat_df[use_cols].astype(np.float32).to_numpy(dtype=np.float32)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    Xs = scaler.transform(X_all).astype(np.float32)
    X_last = _to_seq_last(Xs, T)

    model = tf.keras.models.load_model(model_path)
    ret_hat = float(model.predict(X_last, verbose=0).reshape(-1)[0])

    # sigma résiduelle
    ret1 = pd.to_numeric(raw["Close"], errors="coerce").pct_change().shift(-1)
    y_all = ret1.reindex(feat_df.index).to_numpy(dtype=np.float32)
    sigma = _residual_sigma(model, Xs, y_all, T)

    pred_close = last_close * (1.0 + ret_hat)
    d_pct = (pred_close / last_close) - 1.0

    print(f"{sym} -> last_close={last_close:.2f} pred_close={pred_close:.2f} d_pct={d_pct*100:.2f}")
    if sigma > 0:
        low = last_close * (1.0 + ret_hat - 1.64 * sigma)
        high = last_close * (1.0 + ret_hat + 1.64 * sigma)
        print(f"ret_lstm={ret_hat:.5f} sigma={sigma:.5f} CI90%=[{low:.2f},{high:.2f}]")
    else:
        print(f"ret_lstm={ret_hat:.5f} sigma=0.00000")
    return 0

if __name__ == "__main__":
    sys.exit(main())
