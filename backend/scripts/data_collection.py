# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def fetch_one(symbol: str, period: str):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"Aucune donnée pour {symbol} avec period='{period}'")

    # Certains téléchargements ont des colonnes MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()

    # Normalise les noms de colonnes
    rename = {str(c): str(c).capitalize() for c in df.columns}
    if "Adj close" in rename.values():
        rename = {k: ("Adj Close" if v == "Adj close" else v) for k, v in rename.items()}
    df = df.rename(columns=rename)

    # Colonnes minimales
    for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["Symbol"] = symbol
    df = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    out = DATA / f"{symbol}_historical_prices.csv"
    df.to_csv(out, index=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, help="Un seul ticker, p.ex. AAPL")
    ap.add_argument("--symbols", type=str, help="Liste de tickers séparés par des virgules")
    ap.add_argument("--period", type=str, default="10y", help="Fenêtre yfinance, ex: 1y, 5y, 10y, max")
    ap.add_argument("--years", type=int, help="Alias pratique: N années -> period=f\"{N}y\"")
    args = ap.parse_args()

    if args.years is not None:
        period = f"{args.years}y"
    else:
        period = args.period

    tickers = []
    if args.symbol:
        tickers.append(args.symbol.strip())
    if args.symbols:
        tickers.extend([s.strip() for s in args.symbols.split(",") if s.strip()])

    if not tickers:
        raise SystemExit("Spécifie --symbol AAPL ou --symbols AAPL,NVDA,...")

    for t in tickers:
        p = fetch_one(t, period)
        print(f"Saved: {p}")

if __name__ == "__main__":
    main()
