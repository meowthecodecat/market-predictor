# FILENAME: scripts/data_preprocessing.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def _safe_std(s: pd.Series) -> pd.Series:
    std = s.rolling(20, min_periods=1).std()
    # évite NaN et divisions par zéro
    return std.replace(0.0, np.nan).fillna(1.0)

def make_features(df: pd.DataFrame, time_step: int = 30):
    df = df.copy()

    # cast numérique
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # clean + ordre chronologique
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    high = df.get("High", close)
    low = df.get("Low", close)
    vol = df.get("Volume", pd.Series(index=df.index, dtype=float)).fillna(0.0)

    # retours
    df["ret_1"] = close.pct_change().fillna(0.0)
    df["ret_5"] = close.pct_change(5).fillna(0.0)
    df["ret_10"] = close.pct_change(10).fillna(0.0)

    # range HL
    df["hl_range"] = ((high - low) / close.replace(0, np.nan)).fillna(0.0)

    # moyennes mobiles avec min_periods=1 pour éviter la perte de lignes
    df["sma_5"] = close.rolling(5, min_periods=1).mean()
    df["sma_10"] = close.rolling(10, min_periods=1).mean()
    df["sma_20"] = close.rolling(20, min_periods=1).mean()

    # EMA sans NaN initiaux
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # RSI(14) robustifié
    delta = close.diff().fillna(0.0)
    up = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = up / (down.replace(0.0, np.nan))
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["rsi_14"] = 100 - (100 / (1 + rs.replace(-1.0, 0.0)))

    # ratios
    df["sma5_div_close"] = (df["sma_5"] / close.replace(0, np.nan)).fillna(1.0)
    df["sma20_div_close"] = (df["sma_20"] / close.replace(0, np.nan)).fillna(1.0)
    df["ema12_div_ema26"] = (df["ema_12"] / df["ema_26"].replace(0, np.nan)).fillna(1.0)

    # z-score volume robuste
    vol_mean = vol.rolling(20, min_periods=1).mean()
    vol_std = _safe_std(vol)
    df["vol_z"] = ((vol - vol_mean) / vol_std).fillna(0.0)

    # pas de drop de lignes: on garde la longueur d'entrée (utile pour le test)
    feat_cols = [
        "Close","High","Low","Volume",
        "ret_1","ret_5","ret_10","hl_range",
        "sma_5","sma_10","sma_20","ema_12","ema_26",
        "rsi_14","sma5_div_close","sma20_div_close","ema12_div_ema26","vol_z"
    ]
    # assure la présence des colonnes manquantes si besoin
    for c in ["High","Low","Volume"]:
        if c not in df.columns:
            df[c] = 0.0

    return df, feat_cols

def make_features_from_df(df: pd.DataFrame, time_step: int = 30):
    # alias pour compatibilité tests
    return make_features(df, time_step)
