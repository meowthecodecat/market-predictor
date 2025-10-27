# FILENAME: scripts/evaluate_backtest.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

def load_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    c_date = cols.get("date")
    c_close = cols.get("close")
    if c_date is None or c_close is None:
        raise ValueError(f"Colonnes manquantes dans {csv_path.name}")
    df = df[[c_date, c_close]].rename(columns={c_date: "Date", c_close: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    return df

def stats_on(df: pd.DataFrame) -> dict:
    ret1 = df["Close"].pct_change().shift(-1).dropna()
    return {
        "n_days": int(ret1.shape[0]),
        "avg_ret_1d": float(ret1.mean()),
        "vol_1d": float(ret1.std()),
        "bh_cum": float((1.0 + ret1).prod() - 1.0),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*_historical_prices.csv"))
    if not files:
        print("Backtest: aucun fichier de prix trouvé.")
        return

    print("Backtest sommaire prix (buy&hold, informatif):")
    print("symbol,n_days,avg_ret_1d,vol_1d,bh_cum_or_status")

    for f in files:
        sym = f.name.split("_")[0]
        try:
            df = load_clean(f)
            s = stats_on(df)
            print(f"{sym},{s['n_days']},{s['avg_ret_1d']},{s['vol_1d']},{s['bh_cum']}")
        except Exception as e:
            print(f"{sym},0,,,'FAIL: {e}'")

if __name__ == "__main__":
    main()
