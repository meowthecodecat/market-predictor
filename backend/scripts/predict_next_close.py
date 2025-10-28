# FILENAME: scripts/predict_next_close.py
# -*- coding: utf-8 -*-
"""
Prédiction close J+1 avec ensemblage LSTM + GBM calibré.
- Charge nextclose_{SYM}.keras + scaler + features
- Calcule ret_hat LSTM
- Charge tabular_gbm_{SYM}_calibrated.pkl si dispo → p_up
- Combine: ret_final = ret_lstm * max(0, (p_up - 0.5) * 2)
- Calcule intervalle ±σ (estimé sur historique)
- Imprime: "{SYM} -> last_close=... pred_close=... d_pct=..."

Usage:
  python scripts/predict_next_close.py --symbol AAPL --time-step 30 --alpha 0.6
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

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
    # estime σ des résidus sur l'historique
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
    ap.add_argument("--alpha", type=float, default=0.6)  # non utilisé directement ici
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

    raw = _load_prices(sym)
    # exog si dispos
    def _try(symbol: str) -> pd.DataFrame | None:
        p = DATA / f"{symbol}_historical_prices.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.capitalize() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
        return None
    exog = {}
    for k in ["SPY","XLK","DXY","US10Y"]:
        dfk = _try(k)
        if dfk is not None:
            exog[k] = dfk

    feat_df, _ = make_features(raw, market_df=None, earnings_dates=None, exog=exog)
    last_close = float(raw["Close"].iloc[-1])

    # LSTM
    X_all = feat_df[use_cols].astype(np.float32)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    Xs = scaler.transform(X_all)
    X_last = _to_seq_last(Xs, T)

    model = tf.keras.models.load_model(model_path)
    ret_lstm = float(model.predict(X_last, verbose=0).reshape(-1)[0])

    # σ résiduelle estimation
    # reconstitue y_all (ret_+1) aligné
    ret1 = pd.to_numeric(raw["Close"], errors="coerce").pct_change().shift(-1)
    y_all = ret1.reindex(feat_df.index).to_numpy(dtype=np.float32)
    sigma = _residual_sigma(model, Xs, y_all, T)

    # GBM calibré (optionnel)
    p_up = None
    gbm_file = MODELS / f"tabular_gbm_{sym}_calibrated.pkl"
    if gbm_file.exists():
        obj = joblib.load(gbm_file)
        gbm = obj["model"]
        scal = obj["scaler"]
        fcols = obj["features"]
        X_tab = feat_df[fcols].astype(float).to_numpy()
        X_tab_s = scal.transform(X_tab)
        p_up = float(gbm.predict_proba(X_tab_s[-1:])[:,1][0])

    # ensemblage
    if p_up is None:
        ret_final = ret_lstm
    else:
        # amplifie/diminue la magnitude selon la confiance directionnelle
        conf = max(0.0, (p_up - 0.5) * 2.0)  # 0..1
        ret_final = ret_lstm * conf

    pred_close = last_close * (1.0 + ret_final)
    d_pct = (pred_close / last_close) - 1.0

    # impression attendue par run_all.py
    print(f"{sym} -> last_close={last_close:.2f} pred_close={pred_close:.2f} d_pct={d_pct*100:.2f}")
    # info additionnelle lisible
    if p_up is not None:
        low = last_close * (1.0 + ret_final - 1.64*sigma)
        high = last_close * (1.0 + ret_final + 1.64*sigma)
        print(f"p_up={p_up:.3f}  ret_lstm={ret_lstm:.5f}  ret_final={ret_final:.5f}  CI90%=[{low:.2f},{high:.2f}]")

    return 0

if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
