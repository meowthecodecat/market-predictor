# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=64)
def load_prices(csv_path: str):
    p = Path(csv_path)
    df = pd.read_csv(p)
    df.columns = [c.capitalize() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df
