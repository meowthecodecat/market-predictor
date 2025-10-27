# -*- coding: utf-8 -*-
"""
Prédit le prochain close pour un symbole en réutilisant EXACTEMENT
les features et le scaler sauvegardés à l'entraînement.
Gère aussi les anciens fichiers features.json qui contiennent une simple liste.
"""
from __future__ import annotations

import argparse, json, os, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

# local
sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import make_features_from_df

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

def _load_pickle(p):
    if not Path(p).exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

def _make_last_sequence(X_all: np.ndarray, T: int) -> np.ndarray:
    if len(X_all) < T:
        return np.empty((0, T, X_all.shape[1]), dtype=float)
    return X_all[-T:, :].reshape(1, T, X_all.shape[1])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--time-step", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    sym = args.symbol.upper()
    data_path = Path(args.data_dir) / f"{sym}_historical_prices.csv"
    if not data_path.exists():
        print(f"{sym} -> ERREUR: données introuvables: {data_path}")
        return 0

    # Artefacts
    lstm_path  = Path(args.models_dir) / f"nextclose_{sym}.keras"
    scaler_path = Path(args.models_dir) / f"nextclose_{sym}_scaler.pkl"
    feats_path  = Path(args.models_dir) / f"nextclose_{sym}_features.json"

    if args.debug:
        print(f"{sym} DEBUG -> lstm={lstm_path if lstm_path.exists() else None}  "
              f"tab=None  scaler={scaler_path if scaler_path.exists() else None}")
    if not lstm_path.exists() or not scaler_path.exists() or not feats_path.exists():
        print(f"{sym} -> ERREUR: artefacts manquants (modèle/scaler/features).")
        return 0

    # Charger méta-features. Supporte dict moderne et liste legacy.
    with open(feats_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if isinstance(meta, list):
        use_cols = meta
        T_train = args.time_step
    elif isinstance(meta, dict):
        use_cols = meta.get("features", [])
        T_train  = int(meta.get("time_step", args.time_step))
    else:
        print(f"{sym} -> ERREUR: format features.json invalide.")
        return 0

    # Données + features
    raw = pd.read_csv(data_path)
    raw = raw.rename(columns={c: c.capitalize() for c in raw.columns})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    feat_df, _ = make_features_from_df(raw)

    # Aligner colonnes dans le même ordre que l'entraînement
    missing = [c for c in use_cols if c not in feat_df.columns]
    if len(missing) > 0:
        if args.debug:
            print(f"{sym} DEBUG -> features manquantes: {missing}")
        print(f"{sym} -> ERREUR: mismatch features entre train et prédiction.")
        return 0

    X_all_df = feat_df[use_cols].copy().astype(float)
    scaler = _load_pickle(scaler_path)
    try:
        X_all = scaler.transform(X_all_df)
    except Exception:
        X_all = X_all_df.to_numpy(dtype=float)

    # last close
    if "Close" not in raw.columns:
        print(f"{sym} -> ERREUR: colonne Close manquante.")
        return 0
    last_close = float(pd.to_numeric(raw["Close"], errors="coerce").dropna().iloc[-1])

    # séquence
    X_seq = _make_last_sequence(np.asarray(X_all, dtype=float), T_train)
    if X_seq.size == 0:
        print(f"{sym} -> ERREUR: pas assez d'observations pour time_step={T_train}.")
        return 0

    # modèle
    try:
        lstm = tf.keras.models.load_model(lstm_path, compile=False)
    except Exception:
        print(f"{sym} -> ERREUR: chargement modèle impossible.")
        return 0

    try:
        ret_hat = float(np.ravel(lstm.predict(X_seq, verbose=0))[-1])
    except Exception:
        print(f"{sym} -> ERREUR: prédiction impossible.")
        return 0

    pred_close = last_close * (1.0 + ret_hat)
    d_pct = (pred_close / last_close - 1.0) * 100.0

    if args.debug:
        print(f"{sym} DEBUG -> cols_used={len(use_cols)}  X_all shape={X_all_df.shape}  T={T_train}")

    print(f"{sym} -> last_close={last_close:.2f}  pred_close={pred_close:.2f}  d_pct={d_pct:.2f}%  (alpha={args.alpha:.2f})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
