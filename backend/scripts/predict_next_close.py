# FILENAME: scripts/predict_next_close.py
# -*- coding: utf-8 -*-
"""
Prédit uniquement le close J+1 à partir du LSTM de régression.
- Charge nextclose_{SYM}.keras + scaler + features
- Sort ret_hat LSTM, reconstruit pred_close = last_close * (1 + ret_hat)
- Donne en plus un intervalle 90% via sigma résiduelle (informatif, non bloquant)
- Imprime: "{SYM} -> last_close=... pred_close=... d_pct=..."

Usage:
  python scripts/predict_next_close.py --symbol AAPL --time-step 30
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import make_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

def _load_prices(symbol: str) -> pd.DataFrame:
    p = DATA / f"{symbol}_historical_prices.csv"
    if not p.exists():
        raise FileNotFoundError(f"Données introuvables: {p}")
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

def _to_seq_last(X2: np.ndarray, T: int) -> np.ndarray:
    if len(X2) < T:
        raise ValueError(f"Pas assez d'observations pour T={T} (n={len(X2)})")
    return X2[-T:, :][None, :, :]

def _residual_sigma(model, X_all_s: np.ndarray, y_all: np.ndarray, T: int) -> float:
    xs, ys = [], []
    for i in range(T, len(X_all_s)):
        xs.append(X_all_s[i - T:i, :])
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
    args = ap.parse_args()

    sym = args.symbol.upper()
    T = int(args.time_step)

    # artefacts LSTM
    model_path = MODELS / f"nextclose_{sym}.keras"
    scaler_path = MODELS / f"nextclose_{sym}_scaler.pkl"
    feats_path  = MODELS / f"nextclose_{sym}_features.json"
    if not (model_path.exists() and scaler_path.exists() and feats_path.exists()):
        print(f"{sym} -> modèles/scaler/features manquants")
        return 1

    with open(feats_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    use_cols = meta.get("features", [])
    T = int(meta.get("time_step", T))

    # données
    raw = _load_prices(sym)

    # exog: chargement facultatif si présents en local, sinon ignorés
    def _try(symbol: str) -> pd.DataFrame | None:
        p = DATA / f"{symbol}_historical_prices.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.capitalize() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
        return None
    exog = {k: _try(k) for k in ["SPY","XLK","DXY","US10Y"] if _try(k) is not None}

    feat_df, _ = make_features(raw, market_df=None, earnings_dates=None, exog=exog)
    last_close = float(raw["Close"].iloc[-1])

    # préparation features
    X_all = feat_df[use_cols].astype(np.float32)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    Xs = scaler.transform(X_all)
    X_last = _to_seq_last(Xs, T)

    # modèle
    model = tf.keras.models.load_model(model_path)
    ret_hat = float(model.predict(X_last, verbose=0).reshape(-1)[0])

    # intervalle 90% informatif
    ret1 = pd.to_numeric(raw["Close"], errors="coerce").pct_change().shift(-1)
    y_all = ret1.reindex(feat_df.index).to_numpy(dtype=np.float32)
    sigma = _residual_sigma(model, Xs, y_all, T)

    # close prédit
    pred_close = last_close * (1.0 + ret_hat)
    d_pct = (pred_close / last_close) - 1.0

    print(f"{sym} -> last_close={last_close:.2f} pred_close={pred_close:.2f} d_pct={d_pct*100:.2f}")
    if sigma > 0:
        low = last_close * (1.0 + ret_hat - 1.64*sigma)
        high = last_close * (1.0 + ret_hat + 1.64*sigma)
        print(f"ret_lstm={ret_hat:.5f}  sigma={sigma:.5f}  CI90%=[{low:.2f},{high:.2f}]")
    else:
        print(f"ret_lstm={ret_hat:.5f}  sigma=0.00000")

    return 0

if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
