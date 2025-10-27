# -*- coding: utf-8 -*-
"""
Entraîne un LSTM qui prédit le rendement J+1 puis sauvegarde:
- models/nextclose_{SYM}.keras
- models/nextclose_{SYM}_scaler.pkl
- models/nextclose_{SYM}_features.json  (dict: {"features": [...], "time_step": T})
Compatible TF/Keras où optimizer.learning_rate peut être une chaîne ("auto").
"""
from __future__ import annotations

import argparse, json, os, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Accès local aux utilitaires de features
sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import make_features_from_df

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SET = [
    "ret_1", "ret_5", "ret_10",
    "hl_range", "oc_range",
    "sma5_div_close", "sma20_div_close",
    "ema12_div_ema26",
]

def _safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _build_lstm(input_timesteps: int, input_dim: int, lr: float = 1e-3) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(input_timesteps, input_dim))
    x = tf.keras.layers.LSTM(64, return_sequences=False)(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--time-step", type=int, default=30)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--fine-tune-epochs", type=int, default=12)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    args = p.parse_args()

    sym = args.symbol.upper()
    data_path = Path(args.data_dir) / f"{sym}_historical_prices.csv"
    if not data_path.exists():
        print(f"{sym}: données introuvables: {data_path}")
        return 1

    # 1) Données et features
    raw = pd.read_csv(data_path)
    raw = raw.rename(columns={c: c.capitalize() for c in raw.columns})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    feat_df, _ = make_features_from_df(raw)

    # 2) Colonnes utilisables et cible
    num_df = feat_df.select_dtypes(include=[np.number]).copy()
    if num_df.empty or "Close" not in raw.columns:
        print(f"{sym}: features numériques indisponibles.")
        return 1

    close = pd.to_numeric(raw["Close"], errors="coerce")
    ret1 = close.pct_change().shift(-1)
    target = ret1.reindex(num_df.index)

    # 3) Sélection des features figées
    use_cols = [c for c in FEATURE_SET if c in num_df.columns]
    if len(use_cols) == 0:
        use_cols = [c for c in num_df.columns if c != "Close"][:8]
    X = num_df[use_cols].astype(float).copy()
    y = target.astype(float)

    df = pd.concat([X, y.rename("y")], axis=1).dropna()
    if len(df) < args.time_step + 10:
        print(f"{sym}: pas assez de données après nettoyage.")
        return 1
    X = df[use_cols].copy()
    y = df["y"].copy()

    # 4) Split 80/20 temporel
    n = len(df)
    split = int(n * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    # 5) Scaler
    scaler = StandardScaler()
    scaler.fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # 6) Séquences
    def to_seq(X2, y2, T):
        Xn = np.asarray(X2, dtype=float)
        yn = np.asarray(y2, dtype=float)
        xs, ys = [], []
        for i in range(T, len(Xn)):
            xs.append(Xn[i-T:i, :])
            ys.append(yn[i])
        return np.array(xs), np.array(ys)

    Xtr_seq, ytr_seq = to_seq(Xtr_s, ytr, args.time_step)
    Xte_seq, yte_seq = to_seq(Xte_s, yte, args.time_step)

    # 7) Modèle + fine-tune sans set_value (recompile avec nouveau LR)
    model = _build_lstm(args.time_step, Xtr.shape[1], lr=1e-3)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xte_seq, yte_seq),
        epochs=args.epochs, batch_size=64, verbose=0, callbacks=cb
    )
    if args.fine_tune_epochs > 0:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="mse")
        model.fit(
            Xtr_seq, ytr_seq,
            validation_data=(Xte_seq, yte_seq),
            epochs=args.fine_tune_epochs, batch_size=64, verbose=0, callbacks=cb
        )

    # 8) Éval
    yhat_te = model.predict(Xte_seq, verbose=0).reshape(-1)
    rmse = _safe_rmse(yte_seq, yhat_te)
    mae = float(mean_absolute_error(yte_seq, yhat_te))

    # 9) Sauvegardes
    lstm_path = Path(args.models_dir) / f"nextclose_{sym}.keras"
    scaler_path = Path(args.models_dir) / f"nextclose_{sym}_scaler.pkl"
    feats_path = Path(args.models_dir) / f"nextclose_{sym}_features.json"

    model.save(lstm_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(feats_path, "w", encoding="utf-8") as f:
        json.dump({"features": use_cols, "time_step": args.time_step}, f, ensure_ascii=False, indent=2)

    print(f"{sym}: RMSE(ret J+1)={rmse:.5f}  MAE={mae:.5f}  N_test={len(yte_seq)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
