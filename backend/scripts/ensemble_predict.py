# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

FEATS = ["Open","High","Low","Close","Volume","SMA_14","SMA_50","RSI_14","MACD","MACD_Signal","BB_H","BB_L","Ret_1d","Ret_5d","HL_Range","Vol_Chg"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, default="up_1d", choices=["up_1d","up_5d"])
    ap.add_argument("--alpha", type=float, default=0.5, help="Poids LSTM dans l'agrégation")
    args = ap.parse_args()

    # Dernière fenêtre tabulaire + LSTM du master
    master = pd.read_csv(DATA / "indicators_master.csv").dropna().sort_values("Date")
    X_tab = master[FEATS].values
    scaler_tab = joblib.load(MODELS / f"scaler_tab_{args.label}.pkl")
    X_tab_scaled = scaler_tab.transform(X_tab)

    gbm = joblib.load(ROOT / "market_predictor_gbm.pkl")
    proba_gbm = gbm.predict_proba(X_tab_scaled)[:,1]
    p_gbm = proba_gbm[-1]

    scaler_lstm = joblib.load(MODELS / f"scaler_lstm_{args.label}.pkl")
    X_all_scaled = scaler_lstm.transform(X_tab)
    # utiliser même time-step qu'entraînement => récupérer de X_test_lstm.npy
    ts = np.load(DATA / "X_test_lstm.npy").shape[1]
    X_last = X_all_scaled[-ts:]
    X_last = np.expand_dims(X_last, axis=0)
    lstm = tf.keras.models.load_model(MODELS / f"lstm_{args.label}.keras")
    p_lstm = float(lstm.predict(X_last, verbose=0).ravel()[0])

    p_ens = args.alpha * p_lstm + (1-args.alpha) * p_gbm
    print(f"Proba GBM : {p_gbm:.3f}")
    print(f"Proba LSTM: {p_lstm:.3f}")
    print(f"Ensemble  : {p_ens:.3f}  (alpha={args.alpha})")

if __name__ == "__main__":
    main()
