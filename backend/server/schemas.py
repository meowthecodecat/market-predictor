# backend/server/schemas.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class TickerPred(BaseModel):
    symbol: str
    ts_utc: datetime
    last_close: float
    pred_close: float
    d_pct: float
    status: str
    confidence: float  # 0.0–1.0
