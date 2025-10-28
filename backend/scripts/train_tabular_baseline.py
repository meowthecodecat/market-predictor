# FILENAME: scripts/train_tabular_baseline.py
# -*- coding: utf-8 -*-
"""
Baseline tabulaire directionnelle avec fenêtre récente, pondération temporelle
et walk-forward CV. Compatible sans master.csv.

Usage:
  python scripts/train_tabular_baseline.py --symbol AAPL
  python scripts/train_tabular_baseline.py --csv data/AAPL_historical_prices.csv
  # options:
  #   --lookback-days 540  --half-life-days 90
  #   --start-date 2019-01-01  --step-days 30  --test-days 30

Sorties:
  data/gbm_cv_report.csv
  data/feature_stats_tabular.csv
  models/tabular_gbm.pkl
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# local utils
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import make_features
from utils_data import time_window, time_decay_weights, walk_forward_splits

DEFAULT_FEATURES = [
    # couche structure (existe dans make_features)
    "ret_1","ret_5","ret_10","hl_range",
    "sma5_div_close","sma20_div_close","ema12_div_ema26",
    "rsi_14","vol_z",
    # contexte
    "trend_regime","vol_regime","market_vol","dow","month","month_end_flag","earnings_flag",
    # comportement
    "overnight_gap","intraday_return","gap_fill","volatility_rolling","volume_surge",
]

def _load_prices_from_args(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        p = Path(args.csv)
        df = pd.read_csv(p)
    elif args.symbol:
        sym = args.symbol.upper()
        p = DATA / f"{sym}_historical_prices.csv"
        df = pd.read_csv(p)
    else:
        raise ValueError("Spécifie --symbol SYM ou --csv path.")
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

def _make_label_up1d(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret1 = close.pct_change().shift(-1)
    return (ret1 > 0).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--lookback-days", type=int, default=540)
    ap.add_argument("--half-life-days", type=int, default=90)
    ap.add_argument("--start-date", type=str, default="2019-01-01")
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("--test-days", type=int, default=30)
    args = ap.parse_args()

    # 1) Données
    raw = _load_prices_from_args(args)
    # proxy marché optionnel: ici None par défaut
    feats_df, feat_cols_all = make_features(raw, market_df=None, earnings_dates=None)

    # 2) Fenêtre récente + pondération
    feats_df = time_window(feats_df, lookback_days=args.lookback_days)
    sample_w = time_decay_weights(feats_df, half_life_days=args.half_life_days)

    # 3) Label directionnel
    y = _make_label_up1d(feats_df)

    # 4) Sélection features
    use_cols = [c for c in DEFAULT_FEATURES if c in feats_df.columns]
    if not use_cols:
        # fallback minimal
        use_cols = [c for c in feats_df.columns if c not in ["Date","Close"]][:12]
    X = feats_df[use_cols].astype(float)
    df_xy = pd.concat([feats_df[["Date"]], X, y.rename("y")], axis=1).dropna()

    if len(df_xy) < 400:
        print("Trop peu de données après nettoyage.")
        return

    X = df_xy[use_cols].to_numpy(dtype=float)
    y = df_xy["y"].to_numpy(dtype=int)
    w = sample_w[df_xy.index.to_numpy()]

    # 5) Walk-forward CV
    records = []
    for fold, (tr, va) in enumerate(
        walk_forward_splits(df_xy[["Date"]], start_date=args.start_date,
                            step_days=args.step_days, test_days=args.test_days),
        start=1
    ):
        if len(va) < 20 or len(tr) < 100:
            continue
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X[tr], y[tr], sample_weight=w[tr])

        proba = model.predict_proba(X[va])[:, 1]
        pred = (proba >= 0.5).astype(int)

        records.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "acc": accuracy_score(y[va], pred),
            "prec": precision_score(y[va], pred, zero_division=0),
            "rec": recall_score(y[va], pred, zero_division=0),
            "f1": f1_score(y[va], pred, zero_division=0),
            "auc": roc_auc_score(y[va], proba) if len(np.unique(y[va]))>1 else np.nan,
        })

    rep = pd.DataFrame(records)
    out_report = DATA / "gbm_cv_report.csv"
    rep.to_csv(out_report, index=False)
    print(rep)
    if not rep.empty:
        print("\nMoyennes CV:\n", rep.mean(numeric_only=True))

    # 6) Entraînement final pondéré
    final = GradientBoostingClassifier(random_state=42)
    final.fit(X, y, sample_weight=w)
    joblib.dump(final, MODELS / "tabular_gbm.pkl")
    print(f"\nModèle GBM sauvegardé : {MODELS/'tabular_gbm.pkl'}")
    print(f"Rapport CV : {out_report}")

    # 7) Importances
    feat_stats = pd.DataFrame({"feature": use_cols, "importance": final.feature_importances_}) \
                    .sort_values("importance", ascending=False)
    feat_stats.to_csv(DATA / "feature_stats_tabular.csv", index=False)

if __name__ == "__main__":
    main()
