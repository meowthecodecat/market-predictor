from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
from fastapi import Query

app = FastAPI(title="Market Scanner API")

RUN_SUMMARY = Path(__file__).resolve().parents[1] / "data" / "run_summary.csv"

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/predictions/latest")
def latest():
    if not RUN_SUMMARY.exists():
        raise HTTPException(404, "run_summary.csv introuvable")
    df = pd.read_csv(RUN_SUMMARY)
    # ne retourne que la dernière horodatation
    last_ts = df["ts_utc"].max()
    out = df[df["ts_utc"] == last_ts].to_dict(orient="records")
    return {"ts_utc": last_ts, "rows": out}

@app.get("/predictions/all")
def all_rows(limit: int = 5000):
    if not RUN_SUMMARY.exists():
        raise HTTPException(404, "run_summary.csv introuvable")
    df = pd.read_csv(RUN_SUMMARY).tail(limit)
    return {"rows": df.to_dict(orient="records")}

@app.get("/predictions/history")
def history(symbol: str = Query(...)):
    df = pd.read_csv(RUN_SUMMARY)
    df = df[df["symbol"] == symbol].sort_values("ts_utc")
    return {"symbol": symbol, "rows": df.to_dict(orient="records")}

@app.get("/backtest/latest")
def backtest_latest():
    rep = (Path(__file__).resolve().parents[1] / "data" / "reports")
    files = sorted(rep.glob("backtest_*.csv"))
    if not files: raise HTTPException(404, "aucun backtest")
    df = pd.read_csv(files[-1])
    return {"file": files[-1].name, "rows": df.to_dict(orient="records")}
