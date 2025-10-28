# FILENAME: scripts/predict_next_close.py
# -*- coding: utf-8 -*-
"""
Prédit le close J+1 pour un ticker donné en utilisant le LSTM sauvegardé:
- Charge models/nextclose_{SYM}.keras
- Charge models/nextclose_{SYM}_scaler.pkl
- Charge models/nextclose_{SYM}_features.json
- Recalcule les features, applique lookback si présent, scale, séquence
- Imprime:  "{SYM} -> last_close=... pred_close=... d_pct=..."

Usage:
  python scripts/predict_next_close.py --symbol AAPL --time-step 30 --alpha 0.6
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
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return df

def _to_seq_last(X2: np.ndarray, T: int) -> np.ndarray:
    if len(X2) < T:
        raise ValueError(f"Pas assez d'observations pour T={T} (n={len(X2)})")
    return X2[-T:, :][None, :, :]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--time-step", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.6, help="pondération éventuelle d’un ensemble; ici non utilisé")
    args = ap.parse_args()

    sym = args.symbol.upper()
    # chemins modèles
    model_path = MODELS / f"nextclose_{sym}.keras"
    scaler_path = MODELS / f"nextclose_{sym}_scaler.pkl"
    feats_path  = MODELS / f"nextclose_{sym}_features.json"

    if not model_path.exists() or not scaler_path.exists() or not feats_path.exists():
        print(f"{sym} -> modèles/scaler/features manquants")
        return 1

    with open(feats_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    use_cols = meta.get("features", [])
    T = int(meta.get("time_step", args.time_step))

    # données + features
    raw = _load_prices(sym)
    feat_df, _ = make_features(raw, market_df=None, earnings_dates=None)

    # dernière valeur connue
    last_close = float(raw["Close"].iloc[-1])

    # sélection colonnes
    if not use_cols:
        # fallback: toutes numériques sauf Close
        use_cols = [c for c in feat_df.select_dtypes(include=[np.number]).columns if c != "Close"]

    X = feat_df[use_cols].astype(float).to_numpy()
    # scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    Xs = scaler.transform(X)

    # séquence finale
    X_last = _to_seq_last(Xs, T)

    # modèle
    model = tf.keras.models.load_model(model_path)
    pred_ret = float(model.predict(X_last, verbose=0).reshape(-1)[0])

    # close prédit J+1
    pred_close = last_close * (1.0 + pred_ret)
    d_pct = (pred_close / last_close) - 1.0

    # format EXACT pour run_all.py
    print(f"{sym} -> last_close={last_close:.2f} pred_close={pred_close:.2f} d_pct={d_pct*100:.2f}")
    return 0

if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
