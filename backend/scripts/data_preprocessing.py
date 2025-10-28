# FILENAME: scripts/data_preprocessing.py
# -*- coding: utf-8 -*-
"""
Features structure/contexte/comportement + exogènes + ATR.
API:
  make_features(df, time_step=30, market_df=None, earnings_dates=None, exog=None)
"""
from __future__ import annotations
from typing import Iterable, Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
    return df

def _safe_std(s: pd.Series, win: int = 20) -> pd.Series:
    std = s.rolling(win, min_periods=1).std()
    return std.replace(0.0, np.nan).fillna(1.0)

def _qcut_safe(x: pd.Series, q=3, labels=None) -> pd.Series:
    try:
        return pd.qcut(x, q, labels=labels, duplicates="drop")
    except Exception:
        return pd.cut(x, q, labels=labels)

# -------- Couche 1: Structure --------
def _features_structure(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    d = _ensure_datetime(df.copy()).sort_values("Date").reset_index(drop=True)
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf,-np.inf], np.nan).dropna(subset=["Close"])

    close = d["Close"]
    high = d.get("High", close)
    low = d.get("Low", close)
    vol = d.get("Volume", pd.Series(index=d.index, dtype=float)).fillna(0.0)

    d["ret_1"] = close.pct_change().fillna(0.0)
    d["ret_5"] = close.pct_change(5).fillna(0.0)
    d["ret_10"] = close.pct_change(10).fillna(0.0)
    d["hl_range"] = ((high - low) / close.replace(0, np.nan)).fillna(0.0)

    d["sma_5"] = close.rolling(5, min_periods=1).mean()
    d["sma_10"] = close.rolling(10, min_periods=1).mean()
    d["sma_20"] = close.rolling(20, min_periods=1).mean()
    d["ema_12"] = close.ewm(span=12, adjust=False).mean()
    d["ema_26"] = close.ewm(span=26, adjust=False).mean()

    delta = close.diff().fillna(0.0)
    up = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = (up / down.replace(0.0, np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    d["rsi_14"] = 100 - (100 / (1 + rs.replace(-1.0, 0.0)))

    d["sma5_div_close"] = (d["sma_5"] / close.replace(0, np.nan)).fillna(1.0)
    d["sma20_div_close"] = (d["sma_20"] / close.replace(0, np.nan)).fillna(1.0)
    d["ema12_div_ema26"] = (d["ema_12"] / d["ema_26"].replace(0, np.nan)).fillna(1.0)

    vol_mean = vol.rolling(20, min_periods=1).mean()
    vol_std = _safe_std(vol, 20)
    d["vol_z"] = ((vol - vol_mean) / vol_std).fillna(0.0)

    spread = d["ema_12"] - d["ema_26"]
    d["trend_slope"] = spread.diff(10).fillna(0.0)

    # True Range + ATR_14 pour sizing et SL/TP
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    d["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0.0)
    d["ATR_14"] = d["true_range"].rolling(14, min_periods=1).mean().fillna(method="bfill").fillna(0.0)

    for c in ["High","Low","Volume"]:
        if c not in d.columns: d[c] = 0.0

    cols = [
        "Close","High","Low","Volume",
        "ret_1","ret_5","ret_10","hl_range",
        "sma_5","sma_10","sma_20","ema_12","ema_26",
        "rsi_14","sma5_div_close","sma20_div_close","ema12_div_ema26","vol_z",
        "trend_slope","true_range","ATR_14",
    ]
    return d, cols

# -------- Couche 2: Contexte --------
def _features_context(df: pd.DataFrame,
                      market_df: Optional[pd.DataFrame] = None,
                      earnings_dates: Optional[Iterable[pd.Timestamp]] = None
) -> tuple[pd.DataFrame, List[str]]:
    d = _ensure_datetime(df.copy())
    if "Date" in d.columns:
        d["dow"] = d["Date"].dt.dayofweek.astype("Int8")
        d["month"] = d["Date"].dt.month.astype("Int8")
        d["month_end_flag"] = d["Date"].dt.is_month_end.astype("Int8")
    else:
        d["dow"], d["month"], d["month_end_flag"] = 0, 0, 0

    d["trend_regime"] = np.sign(d["ema_12"] - d["ema_26"]).astype("Int8")
    vol_roll = d["ret_1"].rolling(20, min_periods=1).std().fillna(0.0)
    d["vol_regime"] = _qcut_safe(vol_roll, 3, labels=[0,1,2]).astype("Int8")

    if market_df is not None and {"Date","Close"}.issubset(market_df.columns):
        m = _ensure_datetime(market_df.copy()).sort_values("Date")
        m["mkt_ret_1"] = pd.to_numeric(m["Close"], errors="coerce").pct_change().fillna(0.0)
        m["market_vol"] = m["mkt_ret_1"].rolling(20, min_periods=1).std().fillna(0.0)
        d = pd.merge_asof(d.sort_values("Date"), m[["Date","market_vol"]].sort_values("Date"),
                          on="Date", direction="backward")
        d["market_vol"] = d["market_vol"].fillna(vol_roll)
    else:
        d["market_vol"] = vol_roll

    if earnings_dates is not None:
        edates = {pd.Timestamp(x).normalize() for x in earnings_dates}
        d["earnings_flag"] = d["Date"].dt.normalize().isin(edates).astype("Int8")
    else:
        d["earnings_flag"] = 0

    cols = ["dow","month","month_end_flag","trend_regime","vol_regime","market_vol","earnings_flag"]
    return d, cols

# -------- Couche 3: Comportement --------
def _features_behavior(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    open_ = d.get("Open", d["Close"])
    prev_close = d["Close"].shift(1)

    d["overnight_gap"] = ((open_ - prev_close) / prev_close.replace(0, np.nan)).fillna(0.0)
    d["intraday_return"] = ((d["Close"] - open_) / open_.replace(0, np.nan)).fillna(0.0)
    d["gap_fill"] = (((d["Low"] <= prev_close) & (d["High"] >= prev_close)).astype("Int8")
                     if {"Low","High"}.issubset(d.columns) else 0)
    d["volatility_rolling"] = d["ret_1"].rolling(20, min_periods=1).std().fillna(0.0)

    vol = d.get("Volume", pd.Series(index=d.index, dtype=float)).fillna(0.0)
    vol_ma20 = vol.rolling(20, min_periods=1).mean().replace(0.0, np.nan)
    d["volume_surge"] = (vol / vol_ma20).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    cols = ["overnight_gap","intraday_return","gap_fill","volatility_rolling","volume_surge"]
    return d, cols

# -------- Exogènes --------
def _features_exog(df: pd.DataFrame, exog: Optional[Dict[str, pd.DataFrame]]) -> tuple[pd.DataFrame, List[str]]:
    if not exog: return df, []
    d = df.copy(); feats=[]
    for key, edf in exog.items():
        if edf is None or not {"Date","Close"}.issubset(edf.columns): continue
        e = _ensure_datetime(edf.copy()).sort_values("Date")
        e["exog_ret_1"] = pd.to_numeric(e["Close"], errors="coerce").pct_change().fillna(0.0)
        e = e[["Date","exog_ret_1"]].rename(columns={"exog_ret_1": f"{key}_ret_1"})
        d = pd.merge_asof(d.sort_values("Date"), e.sort_values("Date"), on="Date", direction="backward")
        feats.append(f"{key}_ret_1")
    if "US10Y_ret_1" in d.columns:
        d["rates_shock"] = d["US10Y_ret_1"].rolling(3, min_periods=1).sum()
        feats.append("rates_shock")
    return d, feats

# -------- API --------
def make_features(df: pd.DataFrame,
                  time_step: int = 30,
                  market_df: Optional[pd.DataFrame] = None,
                  earnings_dates: Optional[Iterable[pd.Timestamp]] = None,
                  exog: Optional[Dict[str, pd.DataFrame]] = None
) -> tuple[pd.DataFrame, List[str]]:
    base, cols_base = _features_structure(df)
    ctx, cols_ctx = _features_context(base, market_df, earnings_dates)
    beh, cols_beh = _features_behavior(ctx)
    exo, cols_exo = _features_exog(beh, exog)
    if "Date" in exo.columns: exo = exo.sort_values("Date").reset_index(drop=True)
    return exo, cols_base + cols_ctx + cols_beh + cols_exo

def make_features_from_df(df: pd.DataFrame, time_step: int = 30) -> tuple[pd.DataFrame, List[str]]:
    return make_features(df, time_step=time_step)
