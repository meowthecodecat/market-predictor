# FILENAME: scripts/train_tabular_baseline.py
# -*- coding: utf-8 -*-
from __future__ import annotations
# === DÉTERMINISME TOTAL ===
import os, random, numpy as np
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42); np.random.seed(42)
# ==========================

import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import make_features
from utils_data import time_window, time_decay_weights, chronological_train_test_split

DEFAULT_FEATURES = [
    "ret_1","ret_5","ret_10","hl_range",
    "sma5_div_close","sma20_div_close","ema12_div_ema26","rsi_14","vol_z","trend_slope",
    "trend_regime","vol_regime","market_vol","dow","month","month_end_flag","earnings_flag",
    "overnight_gap","intraday_return","gap_fill","volatility_rolling","volume_surge",
    "SPY_ret_1","XLK_ret_1","DXY_ret_1","US10Y_ret_1","rates_shock",
]

def _load_prices(sym: str) -> pd.DataFrame:
    p = DATA / f"{sym}_historical_prices.csv"
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

def _make_label_up1d(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret1 = close.pct_change().shift(-1)
    return (ret1 > 0).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, required=True)
    ap.add_argument("--lookback-days", type=int, default=540)
    ap.add_argument("--half-life-days", type=int, default=90)
    args = ap.parse_args()

    sym = args.symbol.upper()
    raw = _load_prices(sym)

    def _try(symbol: str):
        p = DATA / f"{symbol}_historical_prices.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.capitalize() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
        return None

    exog = {k: _try(k) for k in ["SPY","XLK","DXY","US10Y"] if _try(k) is not None}

    feats_df, _ = make_features(raw, market_df=None, earnings_dates=None, exog=exog)
    feats_df = time_window(feats_df, lookback_days=args.lookback_days).reset_index(drop=True)
    w = time_decay_weights(feats_df, half_life_days=args.half_life_days)
    w_series = pd.Series(w, index=feats_df.index)

    y = _make_label_up1d(feats_df)
    use_cols = [c for c in DEFAULT_FEATURES if c in feats_df.columns] or \
               [c for c in feats_df.columns if c not in ["Date","Close"]][:12]

    X = feats_df[use_cols].astype(float)
    df_xy = pd.concat([feats_df[["Date"]], X, y.rename("y")], axis=1).dropna()
    if len(df_xy) < 400:
        print("Données insuffisantes.")
        return

    X = df_xy[use_cols].to_numpy(dtype=float)
    y = df_xy["y"].to_numpy(dtype=int)
    w = w_series.loc[df_xy.index].to_numpy(dtype=float)

    tr_idx, va_idx = chronological_train_test_split(df_xy, test_ratio=0.2)
    if len(va_idx) < 50:
        tr_idx, va_idx = np.arange(len(df_xy)-100), np.arange(len(df_xy)-100, len(df_xy))

    scaler = StandardScaler()
    scaler.fit(X[tr_idx])
    Xs = scaler.transform(X)

    base = GradientBoostingClassifier(random_state=42)
    base.fit(Xs[tr_idx], y[tr_idx], sample_weight=w[tr_idx])

    calib = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calib.fit(Xs[va_idx], y[va_idx])

    p = calib.predict_proba(Xs[va_idx])[:,1]
    pred = (p >= 0.5).astype(int)
    rec = {
        "acc": accuracy_score(y[va_idx], pred),
        "prec": precision_score(y[va_idx], pred, zero_division=0),
        "rec": recall_score(y[va_idx], pred, zero_division=0),
        "f1": f1_score(y[va_idx], pred, zero_division=0),
        "auc": roc_auc_score(y[va_idx], p) if len(np.unique(y[va_idx]))>1 else np.nan,
    }
    pd.DataFrame([rec]).to_csv(DATA / f"gbm_cv_report_{sym}.csv", index=False)
    print(rec)

    joblib.dump({"scaler": scaler, "model": base, "features": use_cols, "seed": 42}, MODELS / f"tabular_gbm_{sym}.pkl")
    joblib.dump({"scaler": scaler, "model": calib, "features": use_cols, "seed": 42}, MODELS / f"tabular_gbm_{sym}_calibrated.pkl")
    print(f"Saved: {MODELS / f'tabular_gbm_{sym}.pkl'}")
    print(f"Saved: {MODELS / f'tabular_gbm_{sym}_calibrated.pkl'}")

if __name__ == "__main__":
    main()
