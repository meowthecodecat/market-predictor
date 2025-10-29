# backend/scripts/data_collection.py
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

def _us_session_open(now_utc: datetime) -> bool:
    m = now_utc.hour * 60 + now_utc.minute
    return 13*60+30 <= m < 20*60  # 13:30–20:00 UTC

def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # 1) Aplatir colonnes MultiIndex éventuelles
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if x is not None and str(x) != ""]).strip("_")
            for tup in df.columns.to_list()
        ]
    else:
        df.columns = [str(c) for c in df.columns]

    # 2) Si index datetime, le remettre en colonne
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "index" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"index": "Date"})

    # 3) Déterminer les colonnes candidates par heuristique
    def _find(colkey: str, prefer_adj=False):
        # cherche 'open','high','low','close','volume','date'
        hits = [c for c in df.columns if colkey in c.lower()]
        if not hits:
            return None
        if colkey == "close" and prefer_adj:
            adj = [c for c in hits if "adj" in c.lower()]
            return adj[0] if adj else hits[0]
        return hits[0]

    date_col   = _find("date") or _find("time")
    open_col   = _find("open")
    high_col   = _find("high")
    low_col    = _find("low")
    close_col  = _find("close", prefer_adj=True)  # privilégie Adj Close si présent
    volume_col = _find("volume")

    keep_map = {}
    if date_col:   keep_map[date_col] = "Date"
    if open_col:   keep_map[open_col] = "Open"
    if high_col:   keep_map[high_col] = "High"
    if low_col:    keep_map[low_col] = "Low"
    if close_col:  keep_map[close_col] = "Close"
    if volume_col: keep_map[volume_col] = "Volume"

    if "Close" not in keep_map.values():
        # dernier recours: aucune Close trouvée => vide
        return pd.DataFrame()

    df = df.rename(columns=keep_map)
    df = df[list(keep_map.values())].copy()

    # 4) Typage + nettoyage
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    if "Date" in df.columns:
        df = df.dropna(subset=["Date"]).sort_values("Date")

    return df

def _save_csv(df: pd.DataFrame, path: Path):
    df = _standardize_ohlcv(df)
    if df.empty:
        print(f"[warn] empty after standardize -> skip {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def fetch_daily(symbol: str, years: int):
    start = (datetime.now(timezone.utc) - timedelta(days=365*years)).date().isoformat()
    end = (datetime.now(timezone.utc) - timedelta(days=0)).date().isoformat()
    if _us_session_open(datetime.now(timezone.utc)):
        end = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    df = yf.download(
        symbol,
        interval="1d",
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        repair=True,
        threads=False,
        group_by="column",
    )
    if df is None or df.empty:
        print(f"[warn] no daily data for {symbol}")
        return
    df = df.reset_index()
    _save_csv(df, DATA / f"{symbol}_historical_prices.csv")

def fetch_intraday(symbol: str):
    df_1m = yf.download(
        symbol,
        interval="1m",
        period="7d",
        auto_adjust=True,
        actions=False,
        progress=False,
        repair=True,
        threads=False,
        group_by="column",
    )
    if df_1m is None or df_1m.empty:
        print(f"[warn] no intraday data for {symbol}")
        return
    df_1m = df_1m.reset_index()
    _save_csv(df_1m, DATA / f"{symbol}_intraday_1m.csv")

    # Daily live depuis 1m
    d = _standardize_ohlcv(df_1m)
    if d.empty:
        print(f"[warn] cannot build daily_live for {symbol}")
        return
    d = d.set_index("Date")
    daily = pd.DataFrame({
        "Open":  d["Open"].resample("1D").first(),
        "High":  d["High"].resample("1D").max(),
        "Low":   d["Low"].resample("1D").min(),
        "Close": d["Close"].resample("1D").last(),
        "Volume":d["Volume"].resample("1D").sum(),
    }).dropna(subset=["Open","High","Low","Close"])
    daily.reset_index(inplace=True)
    _save_csv(daily, DATA / f"{symbol}_daily_live.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, required=True)  # ex: "AAPL,NVDA,AMD"
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--intraday", action="store_true")
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for s in syms:
        fetch_daily(s, args.years)
        if args.intraday:
            fetch_intraday(s)

if __name__ == "__main__":
    main()
