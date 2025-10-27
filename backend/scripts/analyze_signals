# -*- coding: utf-8 -*-
import argparse, sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUN_SUMMARY = DATA / "run_summary.csv"
SIGNALS_DIR = DATA / "signals"
PAPER_DIR = DATA / "paper"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)


def cmd_gen(args):
    """Génère des signaux à partir de data/run_summary.csv."""
    src = Path(args.input)
    if not src.exists():
        print(f"Introuvable: {src}")
        sys.exit(1)

    df = pd.read_csv(src)
    if not len(df):
        print("run_summary.csv vide.")
        sys.exit(1)

    if "status" in df.columns:
        df = df[df["status"] == "OK"].copy()
    if not len(df):
        print("Aucune ligne OK dans run_summary.csv.")
        sys.exit(0)

    thr = float(args.threshold)  # seuil en %
    max_abs = float(args.max_abs)

    # Décision
    def decide(p):
        if p >= thr:
            return "long"
        if p <= -thr:
            return "short"
        return "flat"

    df["side"] = df["d_pct"].apply(decide)
    # Taille relative lissée entre 0 et max_abs
    df["size"] = df["d_pct"].abs().clip(0, max_abs) / max_abs
    df.loc[df["side"] == "flat", "size"] = 0.0

    ts_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = SIGNALS_DIR / f"signals_{ts_day}.csv"
    cols = ["symbol", "last_close", "pred_close", "d_pct", "side", "size"]
    df[cols].to_csv(out_path, index=False)

    print(df[cols].to_string(index=False))
    print(f"\nSaved: {out_path}")


def _load_close_series(sym: str) -> pd.Series:
    p = DATA / f"{sym}_historical_prices.csv"
    if not p.exists():
        raise FileNotFoundError(f"Manquant: {p}")
    df = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
    s = df.set_index("Date")["Close"].astype(float)
    return s


def _next_two_trading_points(s: pd.Series, anchor: pd.Timestamp):
    """Retourne (d0, c0), (d1, c1) où d0 = première date >= anchor et d1 = date suivante."""
    idx = s.index
    pos0 = idx.searchsorted(anchor, side="left")
    if pos0 >= len(idx):
        return None, None, None, None
    d0 = idx[pos0]
    if pos0 + 1 >= len(idx):
        return d0, float(s.iloc[pos0]), None, None
    d1 = idx[pos0 + 1]
    return d0, float(s.iloc[pos0]), d1, float(s.iloc[pos0 + 1])


def _latest_signals_file() -> Path:
    files = sorted(SIGNALS_DIR.glob("signals_*.csv"))
    if not files:
        raise FileNotFoundError("Aucun fichier signals_*.csv dans data/signals")
    return files[-1]


def cmd_eval(args):
    """Évalue les signaux sur J+1 (prochaine séance disponible)."""
    if args.signals:
        sig_path = Path(args.signals)
    else:
        sig_path = _latest_signals_file()

    if not sig_path.exists():
        print(f"Introuvable: {sig_path}")
        sys.exit(1)

    sdf = pd.read_csv(sig_path)
    if not len(sdf):
        print("Fichier de signaux vide.")
        sys.exit(0)

    # Déduction de la date d’ancrage à partir du nom de fichier
    # fallback: aujourd’hui UTC
    try:
        stem = sig_path.stem  # signals_YYYY-MM-DD
        ts = stem.split("_")[1]
        anchor = pd.to_datetime(ts)
    except Exception:
        anchor = pd.to_datetime(datetime.now(timezone.utc).date())

    rows = []
    for _, r in sdf.iterrows():
        sym = str(r["symbol"])
        side = str(r["side"]).lower()
        size = float(r.get("size", 0.0))
        if side not in ("long", "short") or size <= 0:
            continue
        try:
            s = _load_close_series(sym)
        except FileNotFoundError as e:
            print(f"Skip {sym}: {e}")
            continue

        d0, c0, d1, c1 = _next_two_trading_points(s, anchor)
        if d0 is None or d1 is None:
            print(f"Skip {sym}: pas assez de points autour de {anchor.date()}")
            continue

        ret_1d = c1 / c0 - 1.0
        pnl = (ret_1d if side == "long" else -ret_1d) * size
        rows.append([sym, side, size, d0.date(), c0, d1.date(), c1, ret_1d, pnl])

    if not rows:
        print("Aucun trade évaluable.")
        out = PAPER_DIR / f"paper_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}.csv"
        pd.DataFrame(
            columns=["symbol", "side", "size", "date_t", "close_t", "date_t1", "close_t1", "ret_1d", "pnl"]
        ).to_csv(out, index=False)
        print(f"Saved: {out}")
        return

    res = pd.DataFrame(
        rows,
        columns=["symbol", "side", "size", "date_t", "close_t", "date_t1", "close_t1", "ret_1d", "pnl"],
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out = PAPER_DIR / f"paper_{stamp}.csv"
    res.to_csv(out, index=False)

    # Stats simples
    agg = {
        "n_trades": len(res),
        "win_rate": float((res["pnl"] > 0).mean()) if len(res) else 0.0,
        "avg_pnl": float(res["pnl"].mean()),
        "sum_pnl": float(res["pnl"].sum()),
        "avg_abs_ret": float(res["ret_1d"].abs().mean()),
    }

    print(res.to_string(index=False))
    print("\nRésumé")
    for k, v in agg.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nSaved: {out}")


def main():
    p = argparse.ArgumentParser(description="Génération et évaluation de signaux.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="Génère les signaux depuis run_summary.csv")
    g.add_argument("--input", default=str(RUN_SUMMARY), help="Chemin du run_summary.csv")
    g.add_argument("--threshold", type=float, default=0.20, help="Seuil décision en % (ex: 0.20)")
    g.add_argument("--max_abs", type=float, default=0.60, help="Cap de |d_pct| pour sizing")
    g.set_defaults(func=cmd_gen)

    e = sub.add_parser("eval", help="Évalue J+1 à partir d’un fichier de signaux")
    e.add_argument("--signals", default="", help="Chemin d’un signals_YYYY-MM-DD.csv (par défaut: dernier)")
    e.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
