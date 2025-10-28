# FILENAME: scripts/utils_data.py
# -*- coding: utf-8 -*-
"""
Utilitaires temporels pour séries financières:
- Chargement CSV avec cache
- Fenêtre récente (lookback)
- Pondération décroissante dans le temps (demi-vie)
- Splits walk-forward chronologiques

Règles:
- Aucune hypothèse sur une fréquence fixe.
- N’altère pas l’ordre d’origine, mais exige une colonne Date.
- Fonctions courtes, avec assertions d’entrées.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Generator, Tuple

import numpy as np
import pandas as pd


# ========= Chargement =========

@lru_cache(maxsize=64)
def load_prices(csv_path: str) -> pd.DataFrame:
    """
    Charge un CSV de prix (colonnes attendues: Date, Open, High, Low, Close, Volume).
    Met Date en datetime et conserve la casse capitalisée.
    """
    p = Path(csv_path)
    assert p.exists(), f"Fichier introuvable: {p}"
    df = pd.read_csv(p)
    df.columns = [c.capitalize() for c in df.columns]
    assert "Date" in df.columns, "Colonne 'Date' requise"
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


# ========= Fenêtrage & pondération =========

def time_window(df: pd.DataFrame, lookback_days: int = 365) -> pd.DataFrame:
    """
    Garde uniquement les lignes dont la Date est dans les 'lookback_days' derniers jours.
    Ne copie pas plus que nécessaire.
    """
    assert "Date" in df.columns, "Colonne 'Date' requise"
    assert lookback_days > 0, "lookback_days doit être > 0"
    if len(df) == 0:
        return df

    max_date = df["Date"].max()
    cutoff = max_date - pd.Timedelta(days=int(lookback_days))
    mask = df["Date"] >= cutoff
    # Conserve l'index et l'ordre
    return df.loc[mask].copy()


def time_decay_weights(df: pd.DataFrame, half_life_days: int = 60) -> np.ndarray:
    """
    Pondère chaque ligne par décroissance exponentielle en fonction de son ancienneté:
        w = 0.5 ** (age_days / half_life_days)
    Normalise pour somme = nombre d’observations (échelle stable).
    """
    assert "Date" in df.columns, "Colonne 'Date' requise"
    assert half_life_days > 0, "half_life_days doit être > 0"
    n = len(df)
    if n == 0:
        return np.asarray([], dtype=float)

    max_date = df["Date"].max()
    age = (max_date - df["Date"]).dt.days.clip(lower=0).astype(float).to_numpy()
    w = np.power(0.5, age / float(half_life_days))
    # Échelle neutre: moyenne ~1
    scale = n / max(w.sum(), 1e-12)
    return (w * scale).astype(float)


# ========= Splits walk-forward =========

def walk_forward_splits(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    step_days: int = 30,
    test_days: int = 30,
    min_train_days: int = 252,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Génère des index (train_idx, val_idx) chronologiques en fenêtre glissante.

    Schéma:
      [---- train ----)|-- test --)
                       +step -> fenêtre suivante

    Paramètres:
      start_date: début du premier bloc (inclus) pour la séparation train/test
      step_days: avance (jours calendaires) entre deux splits successifs
      test_days: taille de la fenêtre de validation (jours calendaires)
      min_train_days: taille minimale d'historique avant la coupure

    Remarques:
      - Utilise les dates réelles pour bâtir les masques (robuste aux jours fériés).
      - S'arrête quand la fenêtre test est vide.
    """
    assert "Date" in df.columns, "Colonne 'Date' requise"
    assert step_days > 0 and test_days > 0 and min_train_days > 0, "Paramètres jours doivent être > 0"
    if len(df) == 0:
        return

    d = df[["Date"]].copy().sort_values("Date").reset_index()
    dates = d["Date"]

    cur = pd.Timestamp(start_date)
    last_date = dates.max()

    # Avance jusqu’à disposer d’au moins min_train_days d’historique
    first_possible_train_start = dates.min()
    earliest_cut = first_possible_train_start + pd.Timedelta(days=min_train_days)
    if cur < earliest_cut:
        cur = earliest_cut

    while True:
        train_end = cur
        test_end = train_end + pd.Timedelta(days=test_days)

        # Masques temporels
        tr_mask = dates < train_end
        va_mask = (dates >= train_end) & (dates < test_end)

        tr_idx = d.loc[tr_mask, "index"].to_numpy()
        va_idx = d.loc[va_mask, "index"].to_numpy()

        if va_idx.size == 0:
            break
        if tr_idx.size == 0:
            # avance jusqu'à trouver un split valide
            cur = cur + pd.Timedelta(days=step_days)
            if cur > last_date:
                break
            continue

        yield tr_idx, va_idx

        cur = cur + pd.Timedelta(days=step_days)
        if cur > last_date:
            break


# ========= Aides simples =========

def chronological_train_test_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split unique 80/20 chronologique par défaut.
    Utile pour des scripts simples ou des baselines.
    """
    assert "Date" in df.columns, "Colonne 'Date' requise"
    assert 0.0 < test_ratio < 1.0, "test_ratio doit être dans (0,1)"
    n = len(df)
    if n < 5:
        return np.arange(0, 0), np.arange(0, 0)
    split = int(n * (1.0 - float(test_ratio)))
    idx = np.arange(n, dtype=int)
    return idx[:split], idx[split:]
