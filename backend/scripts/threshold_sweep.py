# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

def backtest_long_only(returns: np.ndarray, signals: np.ndarray, fees: float=0.0):
    """Perf cumulée simple, sharpe approx, drawdown."""
    trades = signals.astype(bool)
    daily = returns * trades
    # frais sur changement de position
    turns = np.diff(trades.astype(int), prepend=0) != 0
    daily = daily - fees*turns
    eq = (1 + daily).cumprod()
    perf = eq.iloc[-1] - 1 if len(eq)>0 else 0.0
    ret = daily.replace([np.inf,-np.inf], 0).fillna(0)
    sharpe = (ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252) if len(ret)>2 else 0.0
    peak = eq.cummax()
    dd = ((eq/peak)-1).min()
    return float(perf), float(sharpe), float(-dd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, default="up_1d", choices=["up_1d","up_5d"])
    ap.add_argument("--fees", type=float, default=0.0002)
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args()

    # LSTM
    Xte = np.load(DATA / "X_test_lstm.npy")
    yte = np.load(DATA / "y_test_lstm.npy")
    model = tf.keras.models.load_model(MODELS / f"lstm_{args.label}.keras")
    proba = model.predict(Xte, verbose=0).ravel()

    # Retours J+1 alignés sur la même fenêtre que yte via master
    master = pd.read_csv(DATA / "indicators_master.csv").dropna().sort_values("Date")
    close = master["Close"].reset_index(drop=True)
    ret1 = close.pct_change().shift(-1)  # rendement du lendemain
    # réaligner sur X/y séquentiels
    time_step = Xte.shape[1]  # même fenêtre que lors du build
    total_seq = len(master) - time_step
    ret_seq = ret1.iloc[-len(yte):].reset_index(drop=True) if len(ret1) >= len(yte) else pd.Series([0]*len(yte))

    rows = []
    for thr in np.round(np.arange(0.4, 0.91, 0.01), 3):
        pred = (proba >= thr).astype(int)
        acc = accuracy_score(yte, pred)
        prec = precision_score(yte, pred, zero_division=0)
        rec = recall_score(yte, pred, zero_division=0)
        f1 = f1_score(yte, pred, zero_division=0)
        auc = roc_auc_score(yte, proba) if len(np.unique(yte))>1 else np.nan

        if args.long_only:
            signals = pd.Series(pred)
            perf, sharpe, mdd = backtest_long_only(ret_seq.to_frame(0)[0], signals, fees=args.fees)
            rows.append([thr, acc, prec, rec, f1, auc, perf, sharpe, int(signals.sum())])
        else:
            rows.append([thr, acc, prec, rec, f1, auc, np.nan, np.nan, int((proba>=thr).sum())])

    df = pd.DataFrame(rows, columns=["threshold","acc","prec","rec","f1","auc","cum","sharpe","trades"])
    df = df.sort_values(["f1","sharpe"], ascending=False)
    out = DATA / "threshold_sweep_results.csv"
    df.to_csv(out, index=False)

    print("TOP 10 par F1 puis Sharpe :")
    print(df.head(10).to_string(index=False))
    best = df.iloc[0]
    print(f"\nSeuil recommandé: {best['threshold']}")
    print(f"F1={best['f1']:.3f}  Sharpe={best['sharpe']:.2f}  Perf={best['cum']*100:.2f}%  Trades={int(best['trades'])}")

if __name__ == "__main__":
    main()
