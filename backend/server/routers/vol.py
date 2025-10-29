# backend/server/routers/vol.py
from fastapi import APIRouter, Query
from statistics import pstdev
from pathlib import Path
import csv

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/volatility")
def get_volatility(symbol: str = Query("AAPL")):
    # lit un CSV simple ohlc ou returns; ici: colonnes 'date','close'
    path = Path(__file__).resolve().parents[3] / "data" / f"{symbol}_historical_prices.csv"
    closes = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            closes.append(float(row["Close"]))
    rets = [ (closes[i]/closes[i-1]-1.0) for i in range(1, len(closes)) ]
    vol = pstdev(rets) if len(rets) > 5 else 0.0
    return {"symbol": symbol, "vol": vol}
