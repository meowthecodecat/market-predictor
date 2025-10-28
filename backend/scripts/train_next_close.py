# FILENAME: scripts/train_next_close.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import make_features
from utils_data import time_window, time_decay_weights, walk_forward_splits

DEFAULT_FEATURES = [
    "ret_1","ret_5","ret_10","hl_range",
    "sma5_div_close","sma20_div_close","ema12_div_ema26","rsi_14","vol_z",
    "trend_regime","vol_regime","market_vol","dow","month","month_end_flag","earnings_flag",
    "overnight_gap","intraday_return","gap_fill","volatility_rolling","volume_surge",
]

def _safe_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _build_lstm(T: int, D: int, lr: float = 1e-3) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(T, D))
    x = tf.keras.layers.LSTM(64)(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def _to_sequences(X2: np.ndarray, y2: np.ndarray, T: int):
    X2 = np.asarray(X2, dtype=np.float32)
    y2 = np.asarray(y2, dtype=np.float32)
    assert X2.ndim == 2, f"X2 doit être 2D, got {X2.shape}"
    assert y2.ndim == 1, f"y2 doit être 1D, got {y2.shape}"
    xs, ys = [], []
    for i in range(T, len(X2)):
        xs.append(X2[i - T:i, :])
        ys.append(y2[i])
    Xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    if Xs.size == 0:
        return Xs, ys
    assert Xs.ndim == 3, f"Xs doit être 3D (N,T,D), got {Xs.shape}"
    assert ys.ndim == 1 and len(ys) == len(Xs), f"ys alignement invalide: {ys.shape} vs {Xs.shape}"
    return Xs, ys

def _load_prices(data_dir: Path, symbol: str) -> pd.DataFrame:
    p = data_dir / f"{symbol}_historical_prices.csv"
    if not p.exists(): raise FileNotFoundError(f"Données introuvables: {p}")
    raw = pd.read_csv(p)
    raw = raw.rename(columns={c: c.capitalize() for c in raw.columns})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--time-step", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--fine-tune-epochs", type=int, default=12)
    ap.add_argument("--lookback-days", type=int, default=540)
    ap.add_argument("--half-life-days", type=int, default=90)
    ap.add_argument("--start-date", type=str, default="2019-01-01")
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    args = ap.parse_args()

    sym = args.symbol.upper()
    T = int(args.time_step)
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)

    # 1) Données + features
    raw = _load_prices(data_dir, sym)
    feat_df, _ = make_features(raw, market_df=None, earnings_dates=None)

    # 2) Cible ret J+1 alignée
    close = pd.to_numeric(raw["Close"], errors="coerce")
    ret1 = close.pct_change().shift(-1)
    y_full = ret1.reindex(feat_df.index).astype(float)

    # 3) Fenêtre + reset index + poids
    feat_df = time_window(feat_df, args.lookback_days).reset_index(drop=True)
    y_full = y_full.loc[feat_df.index].reset_index(drop=True)

    sample_w = time_decay_weights(feat_df, args.half_life_days)
    w_series = pd.Series(sample_w, index=feat_df.index)

    # 4) Features
    use_cols = [c for c in DEFAULT_FEATURES if c in feat_df.columns]
    if not use_cols:
        use_cols = [c for c in feat_df.select_dtypes(include=[np.number]).columns if c != "Close"][:16]

    X_all_df = feat_df[use_cols].astype(np.float32)
    df_xy = pd.concat([feat_df[["Date"]], X_all_df, y_full.rename("y")], axis=1).dropna()

    if len(df_xy) < T + 50:
        print(f"{sym}: données insuffisantes après nettoyage.")
        return 1

    X_all = df_xy[use_cols].to_numpy(dtype=np.float32)
    y_all = df_xy["y"].to_numpy(dtype=np.float32)
    w_all = w_series.loc[df_xy.index].to_numpy(dtype=np.float32)

    # 5) Walk-forward CV
    cv_records = []
    scaler = StandardScaler()

    for fold, (tr_idx, va_idx) in enumerate(
        walk_forward_splits(df_xy[["Date"]], start_date=args.start_date,
                            step_days=args.step_days, test_days=args.test_days),
        start=1
    ):
        if len(va_idx) < 20 or len(tr_idx) < max(200, T + 50):
            continue

        Xtr, Xva = X_all[tr_idx], X_all[va_idx]
        ytr, yva = y_all[tr_idx], y_all[va_idx]
        wtr = w_all[tr_idx]

        scaler.fit(Xtr)
        Xtr_s = scaler.transform(Xtr).astype(np.float32)
        Xva_s = scaler.transform(Xva).astype(np.float32)

        Xtr_seq, ytr_seq = _to_sequences(Xtr_s, ytr, T)
        Xva_seq, yva_seq = _to_sequences(Xva_s, yva, T)
        if Xtr_seq.size == 0 or Xva_seq.size == 0:
            continue

        # poids alignés
        wtr_seq = wtr[T:].astype(np.float32)
        assert len(wtr_seq) == len(ytr_seq), f"wtr_seq {len(wtr_seq)} != ytr_seq {len(ytr_seq)}"

        # sécurités de shape
        assert Xtr_seq.ndim == 3 and Xtr_seq.shape[1] == T, f"Xtr_seq shape {Xtr_seq.shape}"
        assert Xva_seq.ndim == 3 and Xva_seq.shape[1] == T, f"Xva_seq shape {Xva_seq.shape}"

        model = _build_lstm(T, Xtr_seq.shape[-1], lr=1e-3)
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

        model.fit(
            Xtr_seq, ytr_seq,
            sample_weight=wtr_seq,
            validation_data=(Xva_seq, yva_seq),
            epochs=args.epochs, batch_size=min(64, len(ytr_seq)), verbose=0, callbacks=cb
        )
        if args.fine_tune_epochs > 0:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="mse")
            model.fit(
                Xtr_seq, ytr_seq,
                sample_weight=wtr_seq,
                validation_data=(Xva_seq, yva_seq),
                epochs=args.fine_tune_epochs, batch_size=min(64, len(ytr_seq)), verbose=0, callbacks=cb
            )

        yhat_va = model.predict(Xva_seq, verbose=0).reshape(-1)
        rmse = _safe_rmse(yva_seq, yhat_va)
        mae = float(mean_absolute_error(yva_seq, yhat_va))
        cv_records.append({"fold": fold, "n_tr": int(len(tr_idx)), "n_va": int(len(va_idx)),
                           "rmse": rmse, "mae": mae})

    cv_df = pd.DataFrame(cv_records)
    if not cv_df.empty:
        out_cv = DATA_DIR / f"lstm_cv_{sym}.csv"
        cv_df.to_csv(out_cv, index=False)
        print(cv_df)
        print("\nMoyennes CV:\n", cv_df.mean(numeric_only=True))

    # 6) Entraînement final
    scaler.fit(X_all)
    X_all_s = scaler.transform(X_all).astype(np.float32)
    X_seq, y_seq = _to_sequences(X_all_s, y_all, T)
    if X_seq.size == 0:
        print(f"{sym}: séquences insuffisantes.")
        return 1
    w_seq = w_all[T:].astype(np.float32)
    assert len(w_seq) == len(y_seq), f"w_seq {len(w_seq)} != y_seq {len(y_seq)}"

    final_model = _build_lstm(T, X_seq.shape[-1], lr=1e-3)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    final_model.fit(
        X_seq, y_seq,
        sample_weight=w_seq,
        epochs=args.epochs + max(0, args.fine_tune_epochs // 2),
        batch_size=min(64, len(y_seq)), verbose=0, callbacks=cb
    )

    # 7) Hold-out terminal simple
    n = len(X_all_s)
    split = int(n * 0.8)
    Xtr_s, Xte_s = X_all_s[:split], X_all_s[split:]
    ytr_h, yte_h = y_all[:split], y_all[split:]
    Xtr_h, ytr_hseq = _to_sequences(Xtr_s, ytr_h, T)
    Xte_h, yte_hseq = _to_sequences(Xte_s, yte_h, T)
    if Xte_h.size > 0:
        yhat_te = final_model.predict(Xte_h, verbose=0).reshape(-1)
        rmse_te = _safe_rmse(yte_hseq, yhat_te)
        mae_te = float(mean_absolute_error(yte_hseq, yhat_te))
        print(f"{sym}: RMSE_val_terminal={rmse_te:.5f}  MAE_val_terminal={mae_te:.5f}  N_seq={len(y_seq)}")
    else:
        print(f"{sym}: N_seq={len(y_seq)}")

    # 8) Sauvegardes
    lstm_path = models_dir / f"nextclose_{sym}.keras"
    scaler_path = models_dir / f"nextclose_{sym}_scaler.pkl"
    feats_path = models_dir / f"nextclose_{sym}_features.json"

    final_model.save(lstm_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(feats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "features": use_cols,
                "time_step": T,
                "lookback_days": args.lookback_days,
                "half_life_days": args.half_life_days,
                "start_date": args.start_date,
                "step_days": args.step_days,
                "test_days": args.test_days,
            },
            f, ensure_ascii=False, indent=2
        )
    return 0

if __name__ == "__main__":
    sys.exit(main())
