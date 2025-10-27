# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd, numpy as np, yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVAL_DIR = DATA / "eval"; EVAL_DIR.mkdir(parents=True, exist_ok=True)

def next_trading_close(symbol: str, after_day: date):
    hist = yf.download(symbol, period="90d", interval="1d", auto_adjust=False, progress=False)
    if hist.empty:
        return (None, None)
    hist.index = pd.to_datetime(hist.index).date
    future_idx = [d for d in hist.index if d > after_day]
    if not future_idx:
        return (None, None)
    d1 = min(future_idx)
    return (d1, float(hist.loc[d1, "Close"]))

def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # attend: ts_utc,symbol,pred_close
    if "ts_utc" not in df.columns or "symbol" not in df.columns or "pred_close" not in df.columns:
        raise ValueError("Colonnes requises: ts_utc, symbol, pred_close")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["pred_close"] = pd.to_numeric(df["pred_close"], errors="coerce")
    return df

def main():
    p = argparse.ArgumentParser(description="Évalue run_summary vs close J+1.")
    p.add_argument("--input", default=str(DATA / "run_summary.csv"))
    p.add_argument("--out-dir", default=str(EVAL_DIR))
    p.add_argument("--anchor", type=str,
                   help="Date AAAA-MM-JJ à utiliser comme jour d’ancrage pour TOUS les tickers au lieu de ts_utc.")
    args = p.parse_args()

    run_path = Path(args.input)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_run(run_path)

    anchor_day = None
    if args.anchor:
        try:
            anchor_day = datetime.strptime(args.anchor, "%Y-%m-%d").date()
        except ValueError:
            print("Erreur: --anchor doit être au format AAAA-MM-JJ", file=sys.stderr); sys.exit(1)

    rows = []
    for _, r in df.iterrows():
        sym = r["symbol"]
        pred = r["pred_close"]
        if pd.isna(pred):
            rows.append([sym, r["ts_utc"], np.nan, pred, np.nan, "INVALID_INPUT"]); continue

        base_day = anchor_day if anchor_day else r["ts_utc"].date()
        d1, close1 = next_trading_close(sym, base_day)
        if d1 is None or close1 is None:
            rows.append([sym, r["ts_utc"], np.nan, pred, np.nan, "PENDING"]); continue

        err_pct = (pred - close1) / close1 * 100.0
        rows.append([sym, r["ts_utc"], close1, pred, err_pct, "OK"])

    out = pd.DataFrame(rows, columns=["symbol","ts_utc","real_close_j1","pred_close","err_pct","status"])
    ok = out[out["status"]=="OK"].dropna(subset=["err_pct"])
    mae_pct = float(np.mean(np.abs(ok["err_pct"]))) if len(ok) else np.nan
    rmse_pct = float(np.sqrt(np.mean(ok["err_pct"]**2))) if len(ok) else np.nan

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path_eval = out_dir / f"eval_{stamp}.csv"
    out.to_csv(path_eval, index=False)

    path_log = out_dir / "eval_log.csv"
    log_row = pd.DataFrame([{
        "ts_eval_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "n_ok": int(len(ok)), "n_total": int(len(out)),
        "mae_pct": mae_pct, "rmse_pct": rmse_pct,
        "anchor_used": args.anchor if args.anchor else ""
    }])
    if path_log.exists():
        log = pd.read_csv(path_log); log = pd.concat([log, log_row], ignore_index=True)
    else:
        log = log_row
    log.to_csv(path_log, index=False)

    if len(ok):
        print(f"Évaluation: {path_eval}")
        print(f"OK: {len(ok)}/{len(out)} | MAE%: {mae_pct:.3f}  RMSE%: {rmse_pct:.3f}")
    else:
        print(f"Évaluation: {path_eval}")
        print("Aucune ligne OK (données J+1 non disponibles pour la date choisie).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr); sys.exit(1)
