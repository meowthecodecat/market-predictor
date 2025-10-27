# backend/server/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path

from . import jobs, utils

APP = FastAPI(title="Market Scanner API")
app=APP
# CORS
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class SummaryItem(BaseModel):
    ts_utc: str
    symbol: str
    last_close: float
    pred_close: float
    d_pct: float
    status: str

class RunResponse(BaseModel):
    job_id: str

class RunStatus(BaseModel):
    job_id: str
    running: bool
    exit_code: int | None

class LogsResponse(BaseModel):
    lines: List[str]

class ModelItem(BaseModel):
    symbol: str
    lstm: bool
    scaler: bool
    updated_at: str | None

# Routes
@APP.get("/api/summary", response_model=List[SummaryItem])
def get_summary():
    backend_root = Path(__file__).resolve().parents[1]
    csv_path = backend_root / "data" / "run_summary.csv"
    if not csv_path.exists():
        return []
    return utils.read_summary_csv(csv_path)

@APP.post("/api/run", response_model=RunResponse)
def post_run():
    backend_root = Path(__file__).resol_
