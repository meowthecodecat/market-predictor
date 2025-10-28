# FILENAME: scripts/threshold_sweep.py
# -*- coding: utf-8 -*-
"""
Sweep de seuils AVEC régimes de volatilité et frais variables.

Entrées attendues:
- data/X_test_lstm.npy, data/y_test_lstm.npy
- models/lstm_<label>.keras
- data/indicators_master.csv (colonnes Date, Open, High, Low, Close, Volume)

Sortie:
- data/threshold_sweep_results.csv
Affiche le TOP 10 et un seuil recommandé.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

# ---------- Backtests ----------

def backtest_long_only_var_fees(returns: np.ndarray, signals: np.ndarray, fees_per_day: np.ndarray):
    """Perf cumulée, Sharpe approx, MaxDD, frais variables par jour (bps décimal)."""
    s = signals.astype(int)
    # variations de position (0/1)
    turns = np.diff(s, prepend=0) != 0
    daily = returns * s - fees_per_day * turns
    eq = (1.0 + pd.Series(daily)).cumprod()
    perf = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
    ret = pd.Series(daily).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    sharpe = float((ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252)) if len(ret) > 2 else 0.0
    peak = eq.cummax()
    mdd = float(((eq / peak) - 1.0).min()) if len(eq) else 0.0
    return perf, sharpe, -mdd

# ---------- Régimes ----------

def make_vol_regime(close: pd.Series) -> pd.Series:
    ret1 = close.pct_change().fillna(0.0)
    vol20 = ret1.rolling(20, min_periods=1).std().fillna(0.0)
    # 0=faible,1=moyen,2=fort
    q = pd.qcut(vol20, 3, labels=[0, 1, 2], duplicates="drop")
    return q.astype("Int8")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, default="up_1d", choices=["up_1d", "up_5d"])
    ap.add_argument("--fees", type=float, default=0.0002, help="frais de base par aller-retour (bps décimal)")
    ap.add_argument("--mult-low", type=float, default=1.0, help="multiplicateur frais régime vol=0")
    ap.add_argument("--mult-mid", type=float, default=1.5, help="multiplicateur frais régime vol=1")
    ap.add_argument("--mult-high", type=float, default=2.0, help="multiplicateur frais régime vol=2")
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args()

    # 1) Charger modèle + X/y test séquentiels
    Xte = np.load(DATA / "X_test_lstm.npy")
    yte = np.load(DATA / "y_test_lstm.npy")
    model = tf.keras.models.load_model(MODELS / f"lstm_{args.label}.keras")
    proba = model.predict(Xte, verbose=0).ravel()

    # 2) Charger master prix et construire ret J+1 + régimes alignés
    master = pd.read_csv(DATA / "indicators_master.csv").dropna().sort_values("Date")
    master["Date"] = pd.to_datetime(master["Date"])
    close = pd.to_numeric(master["Close"], errors="coerce")
    ret1 = close.pct_change().shift(-1).fillna(0.0)

    time_step = Xte.shape[1]
    # Aligner la longueur sur yte (même fenêtrage que séquences)
    ret_seq = ret1.iloc[-len(yte):].reset_index(drop=True)
    regime_full = make_vol_regime(close)
    regime_seq = regime_full.iloc[-len(yte):].reset_index(drop=True)

    rows = []
    # frais par régime
    fees_map = {
        0: args.fees * args.mult_low,
        1: args.fees * args.mult_mid,
        2: args.fees * args.mult_high,
    }
    fees_per_day = regime_seq.map(fees_map).astype(float).to_numpy()

    for thr in np.round(np.arange(0.40, 0.91, 0.01), 3):
        pred = (proba >= thr).astype(int)

        # Métriques globales
        acc = accuracy_score(yte, pred)
        prec = precision_score(yte, pred, zero_division=0)
        rec = recall_score(yte, pred, zero_division=0)
        f1 = f1_score(yte, pred, zero_division=0)
        auc = roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else np.nan

        # Backtest global avec frais variables par régime
        signals = pd.Series(pred)
        perf, sharpe, mdd = backtest_long_only_var_fees(
            ret_seq.to_numpy(dtype=float), signals.to_numpy(dtype=int), fees_per_day
        )

        # Comptes par régime (diagnostic)
        trades = int(signals.sum())
        n0 = int((regime_seq == 0).sum())
        n1 = int((regime_seq == 1).sum())
        n2 = int((regime_seq == 2).sum())

        rows.append([
            thr, acc, prec, rec, f1, auc, perf, sharpe, mdd, trades, n0, n1, n2
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "threshold", "acc", "prec", "rec", "f1", "auc",
            "cum_perf", "sharpe", "max_dd", "trades", "n_reg0", "n_reg1", "n_reg2"
        ],
    )
    df = df.sort_values(["f1", "sharpe"], ascending=False)
    out = DATA / "threshold_sweep_results.csv"
    df.to_csv(out, index=False)

    print("TOP 10 par F1 puis Sharpe :")
    print(df.head(10).to_string(index=False))

    best = df.iloc[0]
    print(f"\nSeuil recommandé: {best['threshold']}")
    print(
        f"F1={best['f1']:.3f}  Sharpe={best['sharpe']:.2f}  "
        f"Perf={best['cum_perf']*100:.2f}%  MDD={best['max_dd']*100:.2f}%  Trades={int(best['trades'])}"
    )
    print(
        f"Frais base={args.fees:.4f}  mult(low,mid,high)=({args.mult_low},{args.mult_mid},{args.mult_high})"
    )
    print(f"Résultats: {out}")

if __name__ == "__main__":
    main()
