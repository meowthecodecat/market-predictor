# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, default="up_1d", choices=["up_1d","up_5d"])
    ap.add_argument("--walkforward", type=int, default=5)
    ap.add_argument("--model", type=str, default="gbm", choices=["gbm"])
    args = ap.parse_args()

    master = pd.read_csv(DATA / "indicators_master.csv")
    master = master.dropna().sort_values("Date")
    feats = ["Open","High","Low","Close","Volume","SMA_14","SMA_50","RSI_14","MACD","MACD_Signal","BB_H","BB_L","Ret_1d","Ret_5d","HL_Range","Vol_Chg"]
    X = master[feats].values
    y = master[args.label].values

    scaler = joblib.load(MODELS / f"scaler_tab_{args.label}.pkl")
    X = scaler.transform(X)

    tscv = TimeSeriesSplit(n_splits=args.walkforward)
    records = []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:,1]
        pred = (proba >= 0.5).astype(int)
        records.append({
            "fold": fold,
            "acc": accuracy_score(y[te], pred),
            "prec": precision_score(y[te], pred, zero_division=0),
            "rec": recall_score(y[te], pred, zero_division=0),
            "f1": f1_score(y[te], pred, zero_division=0),
            "auc": roc_auc_score(y[te], proba) if len(np.unique(y[te]))>1 else np.nan
        })

    df_report = pd.DataFrame(records)
    df_report.to_csv(DATA / "gbm_cv_report.csv", index=False)
    print(df_report, "\n\nMoyennes:\n", df_report.mean(numeric_only=True))

    # Entraînement final sur tout le jeu
    final_model = GradientBoostingClassifier(random_state=42)
    final_model.fit(X, y)
    joblib.dump(final_model, ROOT / "market_predictor_gbm.pkl")
    print(f"\nModèle GBM sauvegardé : {ROOT/'market_predictor_gbm.pkl'}")
    print(f"Rapport CV : {DATA/'gbm_cv_report.csv'}")

    # Feature stats
    feat_stats = pd.DataFrame({"feature": feats, "importance": final_model.feature_importances_}).sort_values("importance", ascending=False)
    feat_stats.to_csv(DATA / "feature_stats.csv", index=False)

if __name__ == "__main__":
    main()
