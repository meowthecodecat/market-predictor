# -*- coding: utf-8 -*-
"""
Met à jour:
- data/model_errors.csv : erreurs et hits par date/symbole
- data/preds/preds__<timestamp>.csv : snapshots calibration (p_hat,y_true)

Hypothèses pour p_hat:
- On n'a pas la probabilité native du modèle.
- On convertit l'amplitude de la prédiction (retour prédit) en confiance via une échelle S.
- p_hat = 0.5 + 0.5 * clip(|pred_ret| / S, 0, 1)
- Ajuste S selon ton modèle (SCALE_CONF ci-dessous).
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUN_SUMMARY = DATA / "run_summary.csv"
OUT_ERRORS = DATA / "model_errors.csv"
PREDS_DIR = DATA / "preds"

# Échelle de confiance (retour absolu qui mappe vers p_hat≈1.0)
SCALE_CONF = 0.02  # 2% de move => confiance max. Ajuste si besoin.

def _load_prices(sym: str) -> pd.DataFrame:
    """Charge l'historique d'un symbole et normalise les colonnes."""
    p = DATA / f"{sym}_historical_prices.csv"
    if not p.exists():
        return pd.DataFrame(columns=["Date","Close"])
    df = pd.read_csv(p)
    df.columns = [c.capitalize() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return df[["Date","Close"]]

def _calc_p_hat(pred_ret: float) -> float:
    """Mappe l'amplitude du retour prédit vers une proba d'avoir raison."""
    conf = min(max(abs(float(pred_ret)) / max(SCALE_CONF, 1e-9), 0.0), 1.0)
    return 0.5 + 0.5 * conf

def main() -> int:
    if not RUN_SUMMARY.exists():
        return 0

    rs = pd.read_csv(RUN_SUMMARY)
    need_cols = {"ts_utc","symbol","last_close","pred_close"}
    if not need_cols.issubset(rs.columns):
        return 0

    # Prépare run_summary
    rs = rs.copy()
    rs["ts_utc"] = pd.to_datetime(rs["ts_utc"], errors="coerce")
    rs = rs.dropna(subset=["ts_utc","symbol","last_close","pred_close"])
    rs["pred_ret"] = rs["pred_close"].astype(float) / rs["last_close"].astype(float) - 1.0
    rs["ts_date"] = rs["ts_utc"].dt.normalize()

    # Calcul erreurs + hits par ligne
    err_rows = []
    calib_rows = []

    for sym, grp in rs.groupby("symbol"):
        prices = _load_prices(sym)
        if prices.empty:
            continue
        close = prices["Close"].to_numpy()
        dates = prices["Date"]

        for _, r in grp.iterrows():
            ts_date = r["ts_date"]
            # D = dernière date <= ts_date
            idx = dates.searchsorted(ts_date, side="right") - 1
            if idx < 0 or idx >= len(dates) - 1:
                continue

            last_close_run = float(close[idx])
            actual_close = float(close[idx + 1])
            actual_ret = actual_close / last_close_run - 1.0
            pred_ret = float(r["pred_ret"])

            err_1 = actual_ret - pred_ret
            hit = 1 if (np.sign(actual_ret) == np.sign(pred_ret) and pred_ret != 0.0) else 0

            # Alimentation des deux sorties
            err_rows.append({
                "Date": dates.iloc[idx],
                "Symbol": sym,
                "pred_ret": pred_ret,
                "actual_ret": actual_ret,
                "err_1": err_1,
                "abs_err_1": abs(err_1),
                "hit": hit,
            })

            # Snapshot calibration
            p_hat = _calc_p_hat(pred_ret)
            y_true = int(hit)
            calib_rows.append({"p_hat": p_hat, "y_true": y_true})

    # Rien à écrire
    if not err_rows:
        return 0

    # ===== model_errors.csv =====
    df_err = pd.DataFrame(err_rows).sort_values(["Symbol","Date"]).reset_index(drop=True)

    # Merge incrémental si existant
    if OUT_ERRORS.exists():
        old = pd.read_csv(OUT_ERRORS, parse_dates=["Date"])
        all_df = pd.concat([old, df_err], ignore_index=True)\
                   .drop_duplicates(subset=["Date","Symbol"], keep="last")\
                   .sort_values(["Symbol","Date"]).reset_index(drop=True)
        df_err = all_df

    # Rolling metrics par symbol
    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").reset_index(drop=True)
        g["err_ma5"]  = g["err_1"].rolling(5,  min_periods=1).mean()
        g["err_ma20"] = g["err_1"].rolling(20, min_periods=1).mean()
        g["bias20"]   = g["err_1"].rolling(20, min_periods=1).mean()
        g["hit20"]    = g["hit"].rolling(20, min_periods=1).mean()
        return g

    df_err = df_err.groupby("Symbol", group_keys=False).apply(_roll)
    df_err.to_csv(OUT_ERRORS, index=False)
    print(f"Updated: {OUT_ERRORS} ({len(df_err)} rows)")

    # ===== preds/preds__<ts>.csv =====
    if calib_rows:
        PREDS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        df_calib = pd.DataFrame(calib_rows)[["p_hat","y_true"]]
        df_calib.to_csv(PREDS_DIR / f"preds__{ts}.csv", index=False)
        print(f"Saved calibration snapshot: {PREDS_DIR / f'preds__{ts}.csv'} ({len(df_calib)} rows)")

    return 0

if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
