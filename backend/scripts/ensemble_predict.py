# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# === compatibilité hyper_sweep ===
parser = argparse.ArgumentParser()
parser.add_argument("--symbols", type=str, default="AAPL")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--time-step", type=int, default=20)
parser.add_argument("--label", type=str, choices=["up_1d", "up_5d"], default="up_1d")
parser.add_argument("--alpha", type=float, default=0.5)
args = parser.parse_args()

SYMBOLS = args.symbols.split(",")
THRESHOLD = args.threshold
TIME_STEP = args.time_step
LABEL = args.label
ALPHA = args.alpha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

FEATS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_14", "SMA_50", "RSI_14", "MACD", "MACD_Signal",
    "BB_H", "BB_L", "Ret_1d", "Ret_5d", "HL_Range", "Vol_Chg"
]

def _build_master_if_missing():
    out = DATA / "indicators_master.csv"
    if out.exists():
        return out

    import pandas as pd
    import numpy as np

    frames = []
    for sym in SYMBOLS:
        p = DATA / f"{sym}_historical_prices.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["Date"])
        df = df.sort_values("Date").dropna()

        # Features
        df["SMA_14"] = df["Close"].rolling(14).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()

        delta = df["Close"].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up / (down.replace(0, np.nan))
        df["RSI_14"] = 100 - (100 / (1 + rs))

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        mid = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        df["BB_H"] = mid + 2 * std
        df["BB_L"] = mid - 2 * std

        df["Ret_1d"] = df["Close"].pct_change()
        df["Ret_5d"] = df["Close"].pct_change(5)
        df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]
        df["Vol_Chg"] = df["Volume"].pct_change()

        df["Symbol"] = sym
        frames.append(df[["Date","Symbol","Open","High","Low","Close","Volume",
                          "SMA_14","SMA_50","RSI_14","MACD","MACD_Signal",
                          "BB_H","BB_L","Ret_1d","Ret_5d","HL_Range","Vol_Chg"]])

    if not frames:
        raise FileNotFoundError("Aucun CSV de prix trouvé pour construire indicators_master.csv")

    master = pd.concat(frames, ignore_index=True).dropna().sort_values("Date")
    master.to_csv(out, index=False)
    return out


def main():
    # Assure la présence du master
    master_path = _build_master_if_missing()

    master = pd.read_csv(master_path).dropna().sort_values("Date")
    X_tab = master[FEATS].values

    scaler_tab = joblib.load(MODELS / f"scaler_tab_{LABEL}.pkl")
    X_tab_scaled = scaler_tab.transform(X_tab)

    gbm = joblib.load(ROOT / "market_predictor_gbm.pkl")
    proba_gbm = gbm.predict_proba(X_tab_scaled)[:, 1]
    p_gbm = proba_gbm[-1]

    scaler_lstm = joblib.load(MODELS / f"scaler_lstm_{LABEL}.pkl")
    X_all_scaled = scaler_lstm.transform(X_tab)
    ts = np.load(DATA / "X_test_lstm.npy").shape[1]
    X_last = X_all_scaled[-ts:]
    X_last = np.expand_dims(X_last, axis=0)
    lstm = tf.keras.models.load_model(MODELS / f"lstm_{LABEL}.keras")
    p_lstm = float(lstm.predict(X_last, verbose=0).ravel()[0])

    p_ens = ALPHA * p_lstm + (1 - ALPHA) * p_gbm
    print(f"Proba GBM : {p_gbm:.3f}")
    print(f"Proba LSTM: {p_lstm:.3f}")
    print(f"Ensemble  : {p_ens:.3f}  (alpha={ALPHA})")

