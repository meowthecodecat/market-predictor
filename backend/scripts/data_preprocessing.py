# FILENAME: scripts/data_preprocessing.py
# -*- coding: utf-8 -*-
"""
Feature engineering en 3 couches: structure, contexte, comportement.
- Structure: retours, tendances, volatilité locale, volume.
- Contexte: régimes de tendance/volatilité, calendrier, proxy marché.
- Comportement: ouverture, gap-fill, sursaut de volume, intraday.

API:
    make_features(df, time_step=30, market_df=None, earnings_dates=None)
    make_features_from_df(df, time_step=30)  # alias compat

Notes:
- Ne droppe pas de lignes.
- Tolerant aux colonnes manquantes (Open/High/Low/Volume).
- Sans dépendance externe (pas de TA-Lib).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Optional, Tuple, List


# ========= Utils =========

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


# ========= Couche 1: Structure =========

def _features_structure(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()

    # Cast numérique minimal
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Nettoyage basique
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    d = _ensure_datetime(d)
    if "Date" in d.columns:
        d = d.sort_values("Date").reset_index(drop=True)

    close = d["Close"]
    high = d.get("High", close)
    low = d.get("Low", close)
    vol = d.get("Volume", pd.Series(index=d.index, dtype=float)).fillna(0.0)

    # Retours
    d["ret_1"] = close.pct_change().fillna(0.0)
    d["ret_5"] = close.pct_change(5).fillna(0.0)
    d["ret_10"] = close.pct_change(10).fillna(0.0)

    # Range HL relatif
    d["hl_range"] = ((high - low) / close.replace(0, np.nan)).fillna(0.0)

    # Moyennes mobiles
    d["sma_5"] = close.rolling(5, min_periods=1).mean()
    d["sma_10"] = close.rolling(10, min_periods=1).mean()
    d["sma_20"] = close.rolling(20, min_periods=1).mean()

    # EMA
    d["ema_12"] = close.ewm(span=12, adjust=False).mean()
    d["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # RSI(14) robuste
    delta = close.diff().fillna(0.0)
    up = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = up / down.replace(0.0, np.nan)
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["rsi_14"] = 100 - (100 / (1 + rs.replace(-1.0, 0.0)))

    # Ratios de tendance
    d["sma5_div_close"] = (d["sma_5"] / close.replace(0, np.nan)).fillna(1.0)
    d["sma20_div_close"] = (d["sma_20"] / close.replace(0, np.nan)).fillna(1.0)
    d["ema12_div_ema26"] = (d["ema_12"] / d["ema_26"].replace(0, np.nan)).fillna(1.0)

    # Volume z-score
    vol_mean = vol.rolling(20, min_periods=1).mean()
    vol_std = _safe_std(vol, win=20)
    d["vol_z"] = ((vol - vol_mean) / vol_std).fillna(0.0)

    # Garantit présence colonnes
    for c in ["High", "Low", "Volume"]:
        if c not in d.columns:
            d[c] = 0.0

    cols = [
        "Close", "High", "Low", "Volume",
        "ret_1", "ret_5", "ret_10", "hl_range",
        "sma_5", "sma_10", "sma_20", "ema_12", "ema_26",
        "rsi_14", "sma5_div_close", "sma20_div_close", "ema12_div_ema26", "vol_z",
    ]
    return d, cols


# ========= Couche 2: Contexte =========

def _features_context(df: pd.DataFrame,
                      market_df: Optional[pd.DataFrame] = None,
                      earnings_dates: Optional[Iterable[pd.Timestamp]] = None
                      ) -> Tuple[pd.DataFrame, List[str]]:
    d = _ensure_datetime(df.copy())

    # Calendrier
    if "Date" in d.columns:
        d["dow"] = d["Date"].dt.dayofweek.astype("Int8")  # 0=lundi
        d["month"] = d["Date"].dt.month.astype("Int8")
        d["month_end_flag"] = d["Date"].dt.is_month_end.astype("Int8")
    else:
        d["dow"] = 0
        d["month"] = 0
        d["month_end_flag"] = 0

    # Régimes de tendance/vol
    d["trend_regime"] = np.sign(d["ema_12"] - d["ema_26"]).astype("Int8")
    vol_roll = d["ret_1"].rolling(20, min_periods=1).std().fillna(0.0)
    d["vol_regime"] = _qcut_safe(vol_roll, q=3, labels=[0, 1, 2]).astype("Int8")

    # Proxy de volatilité marché
    # Si market_df fourni (colonnes Date, Close), on calcule la vol rolling du marché et on merge.
    if market_df is not None and {"Date", "Close"}.issubset(set(market_df.columns)):
        m = market_df.copy()
        m = _ensure_datetime(m)
        m = m.sort_values("Date")
        m["mkt_ret_1"] = m["Close"].pct_change().fillna(0.0)
        m["market_vol"] = m["mkt_ret_1"].rolling(20, min_periods=1).std().fillna(0.0)
        d = pd.merge_asof(d.sort_values("Date"), m[["Date", "market_vol"]].sort_values("Date"),
                          on="Date", direction="backward")
        d["market_vol"] = d["market_vol"].fillna(d["ret_1"].rolling(20, min_periods=1).std())
    else:
        # fallback: utilise la vol du titre comme proxy
        d["market_vol"] = d["ret_1"].rolling(20, min_periods=1).std().fillna(0.0)

    # Flag résultats (earnings) si liste fournie
    if earnings_dates is not None:
        earnings_idx = pd.Series(0, index=d.index, dtype="Int8")
        edates = {pd.Timestamp(x).normalize() for x in earnings_dates}
        if "Date" in d.columns:
            earnings_idx = d["Date"].dt.normalize().isin(edates).astype("Int8")
        d["earnings_flag"] = earnings_idx
    else:
        d["earnings_flag"] = 0

    cols = ["dow", "month", "month_end_flag", "trend_regime", "vol_regime", "market_vol", "earnings_flag"]
    return d, cols


# ========= Couche 3: Comportement / Flux =========

def _features_behavior(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    open_ = d.get("Open", d["Close"])

    # Overnight gap (ouverture vs close-1)
    prev_close = d["Close"].shift(1)
    d["overnight_gap"] = ((open_ - prev_close) / prev_close.replace(0, np.nan)).fillna(0.0)

    # Intraday return (close vs open)
    d["intraday_return"] = ((d["Close"] - open_) / open_.replace(0, np.nan)).fillna(0.0)

    # Gap-fill: le prix du jour a-t-il touché le close-1 ?
    d["gap_fill"] = (((d["Low"] <= prev_close) & (d["High"] >= prev_close)).astype("Int8")
                     if {"Low", "High"}.issubset(d.columns) else 0)

    # Volatilité roulante simple
    d["volatility_rolling"] = d["ret_1"].rolling(20, min_periods=1).std().fillna(0.0)

    # Sursaut de volume
    vol = d.get("Volume", pd.Series(index=d.index, dtype=float)).fillna(0.0)
    vol_ma20 = vol.rolling(20, min_periods=1).mean().replace(0.0, np.nan)
    d["volume_surge"] = (vol / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cols = ["overnight_gap", "intraday_return", "gap_fill", "volatility_rolling", "volume_surge"]
    return d, cols


# ========= API principale =========

def make_features(df: pd.DataFrame,
                  time_step: int = 30,
                  market_df: Optional[pd.DataFrame] = None,
                  earnings_dates: Optional[Iterable[pd.Timestamp]] = None
                  ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construit l'ensemble des features. Compatible avec l'ancien appel (df, time_step).
    Args:
        df: DataFrame du titre (Date, Open, High, Low, Close, Volume).
        time_step: réservé pour compat LSTM; non utilisé ici.
        market_df: DataFrame marché (Date, Close) pour proxy volatilité globale.
        earnings_dates: liste/iterable de dates d'earnings à flagger.
    Returns:
        df_features, feat_cols
    """
    base, cols_base = _features_structure(df)
    ctx, cols_ctx = _features_context(base, market_df=market_df, earnings_dates=earnings_dates)
    beh, cols_beh = _features_behavior(ctx)

    feat_cols = cols_base + cols_ctx + cols_beh

    # Ordre chronologique garanti
    if "Date" in beh.columns:
        beh = beh.sort_values("Date").reset_index(drop=True)

    return beh, feat_cols


def make_features_from_df(df: pd.DataFrame, time_step: int = 30) -> Tuple[pd.DataFrame, List[str]]:
    """Alias historique."""
    return make_features(df, time_step=time_step)
