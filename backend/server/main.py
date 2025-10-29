# backend/server/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
from statistics import pstdev
from datetime import datetime, timedelta
import csv

# === Paths ===
SERVER = Path(__file__).resolve()
BACKEND = SERVER.parent.parent       # ../backend
DATA = BACKEND / "data"              # ../backend/data

# === App & CORS ===
app = FastAPI(title="Market Scanner API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://10.146.9.49:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Schemas ===
class SummaryItem(BaseModel):
    ts_utc: str
    symbol: str
    last_close: float
    pred_close: float
    d_pct: float
    status: str
    confidence: Optional[float] = None

# === Helpers ===
def read_summary_csv() -> List[Dict[str, Any]]:
    fp = DATA / "run_summary.csv"
    if not fp.exists():
        return []
    out = []
    with fp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.append({
                    "ts_utc": row.get("ts_utc", ""),
                    "symbol": row.get("symbol", ""),
                    "last_close": float(row.get("last_close") or 0),
                    "pred_close": float(row.get("pred_close") or 0),
                    "d_pct": float(row.get("d_pct") or 0),
                    "status": row.get("status", "OK"),
                    "confidence": float(row.get("confidence")) if "confidence" in (reader.fieldnames or []) and row.get("confidence") else None
                })
            except Exception:
                continue
    return out

def load_closes(symbol: str) -> Dict[str, float]:
    fp = DATA / f"{symbol}_historical_prices.csv"
    m = {}
    if not fp.exists():
        return m
    with fp.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        dkey = "Date" if "Date" in (r.fieldnames or []) else "date"
        ckey = "Close" if "Close" in (r.fieldnames or []) else "close"
        for row in r:
            try:
                m[str(row[dkey])[:10]] = float(row[ckey])
            except Exception:
                continue
    return m

# === Routes ===
@app.get("/api/summary", response_model=List[SummaryItem])
def api_summary():
    return read_summary_csv()

@app.get("/metrics/calibration")
def metrics_accuracy(symbol: Optional[str] = Query(None)):
    # chemins robustes depuis backend/server/main.py
    DATA = Path(__file__).resolve().parent.parent / "data"
    runsum_fp = DATA / "run_summary.csv"

    if not runsum_fp.exists():
        return []

    # 1) charge run_summary
    rows: List[Dict[str, Any]] = []
    with runsum_fp.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if symbol and row.get("symbol") != symbol:
                continue
            try:
                ts = datetime.fromisoformat(row["ts_utc"])
                rows.append({
                    "ts": ts,
                    "day": ts.date().isoformat(),
                    "symbol": row["symbol"],
                    "last_close": float(row["last_close"]),
                    "pred_close": float(row["pred_close"]),
                })
            except Exception:
                continue

    if not rows:
        return []

    # 2) garde le DERNIER run du jour par symbole
    latest_per_day_sym: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        key = (r["day"], r["symbol"])
        if key not in latest_per_day_sym or r["ts"] > latest_per_day_sym[key]["ts"]:
            latest_per_day_sym[key] = r
    latest_rows = list(latest_per_day_sym.values())

    # 3) charge historiques par symbole
    def load_closes(sym: str) -> Dict[str, float]:
        fp = DATA / f"{sym}_historical_prices.csv"
        m: Dict[str, float] = {}
        if not fp.exists():
            return m
        with fp.open(newline="", encoding="utf-8") as f:
            rr = csv.DictReader(f)
            dkey = "Date" if "Date" in (rr.fieldnames or []) else "date"
            ckey = "Close" if "Close" in (rr.fieldnames or []) else "close"
            for row in rr:
                try:
                    m[str(row[dkey])[:10]] = float(row[ckey])
                except Exception:
                    continue
        return m

    closes_cache: Dict[str, Dict[str, float]] = {}
    for r in latest_rows:
        sym = r["symbol"]
        if sym not in closes_cache:
            closes_cache[sym] = load_closes(sym)

    # 4) trouve le PROCHAIN jour de bourse >= date du run
    def next_trading_close(clmap: Dict[str, float], base_date: datetime.date) -> Optional[float]:
        # essaye de base_date+1, puis +2..+7
        for i in range(1, 8):
            d = (base_date + timedelta(days=i)).isoformat()
            if d in clmap:
                return clmap[d]
        # sinon, prend la plus petite date > base_date
        for d in sorted(clmap.keys()):
            try:
                if datetime.fromisoformat(d).date() > base_date:
                    return clmap[d]
            except Exception:
                continue
        return None

    # 5) compute ok par jour
    bucket: Dict[str, List[int]] = {}
    for r in latest_rows:
        sym = r["symbol"]
        clmap = closes_cache.get(sym, {})
        act_c = next_trading_close(clmap, r["ts"].date())
        if act_c is None:
            continue
        pred_up = r["pred_close"] >= r["last_close"]
        real_up = act_c >= r["last_close"]
        ok = 1 if pred_up == real_up else 0
        bucket.setdefault(r["day"], []).append(ok)

    # 6) agrège en accuracy journalière
    series = [
        {"date": day, "accuracy": round(sum(oks) / max(1, len(oks)), 4)}
        for day, oks in sorted(bucket.items())
    ]
    return series

@app.get("/metrics/volatility")
def metrics_volatility(symbol: str = Query("AAPL")):
    fp = DATA / f"{symbol}_historical_prices.csv"
    if not fp.exists():
        raise HTTPException(status_code=404, detail="historical CSV not found")
    closes: List[float] = []
    with fp.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        ck = "Close" if "Close" in r.fieldnames else "close"
        for row in r:
            try:
                closes.append(float(row[ck]))
            except Exception:
                continue
    if len(closes) < 3:
        return {"symbol": symbol, "vol": 0.0}
    rets = [(closes[i] / closes[i-1] - 1.0) for i in range(1, len(closes))]
    return {"symbol": symbol, "vol": pstdev(rets) if len(rets) > 2 else 0.0}
