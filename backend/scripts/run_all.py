# backend/scripts/run_all.py
# -*- coding: utf-8 -*-
"""
Orchestration live + pondération récente + overnight.
Ordre:
1) data_collection (daily + intraday -> daily_live)
2) evaluate_predictions -> MAJ data/model_errors.csv
3) train_next_close + train_tabular_baseline
4) threshold_sweep, evaluate_backtest
5) predict_next_close (priorise *_daily_live.csv)
"""
import subprocess, sys, csv, re, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd  # pour last_open

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

RUN_SUMMARY = DATA / "run_summary.csv"
RUN_LOG = REPORTS / "run_log.csv"

# === Univers
TICKERS = ["AAPL","NVDA","AMD","TSLA","AMZN","GOOGL","MSFT","KO"]
FREEZE_DATA = False  # live

# === Hyperparams par défaut, avec biais "récent"
HP = {
    "LOOKBACK_DAYS": 720,
    "HALF_LIFE_DAYS": 30,
    "START_DATE": "2018-01-01",
    "STEP_DAYS": 30,
    "TEST_DAYS": 30,
    "TIME_STEP": 30,
    "EPOCHS": 20,
    "FINE_TUNE_EPOCHS": 12,
    "ALPHA": 0.60,
    "FEES": 0.0002,
}

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> tuple[str, int]:
    try:
        res = subprocess.run(cmd, cwd=cwd or ROOT, text=True, capture_output=capture, check=True)
        return (res.stdout if capture else ""), 0
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "")
        return (out if capture else ""), e.returncode

def append_run_log(**fields) -> None:
    header_needed = not RUN_LOG.exists()
    keys = list(fields.keys())
    with RUN_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if header_needed: w.writeheader()
        w.writerow(fields)

def get_last_open(symbol: str) -> float:
    live = DATA / f"{symbol}_daily_live.csv"
    base = DATA / f"{symbol}_historical_prices.csv"
    p = live if live.exists() else base
    df = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
    return float(df["Open"].iloc[-1])

def write_run_summary(rows: list[list[str]]) -> None:
    header = ["ts_utc","symbol","last_open","last_close","pred_close","d_pct","status"]
    recreate = False
    if RUN_SUMMARY.exists():
        try:
            with RUN_SUMMARY.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
                recreate = ("last_open" not in first)
        except Exception:
            recreate = True
    mode = "w" if recreate or not RUN_SUMMARY.exists() else "a"
    with RUN_SUMMARY.open(mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(header)
        w.writerows(rows)

def main():
    ts = now_utc()
    append_run_log(ts_utc=ts, step="hparams", status="OK", hp=json.dumps(HP))

    # 1) Collecte
    if not FREEZE_DATA:
        out, rc = run([
            sys.executable, str(SCRIPTS / "data_collection.py"),
            "--symbols", ",".join(TICKERS),
            "--years", "10",
            "--intraday"
        ], capture=True)
        append_run_log(ts_utc=ts, step="collect", status="OK" if rc==0 else f"FAIL({rc})", stdout=out[-2000:])
    else:
        append_run_log(ts_utc=ts, step="collect", status="SKIPPED")

    # 2) Évaluation prédictions passées
    out, rc = run([sys.executable, str(SCRIPTS / "evaluate_predictions.py")], capture=True)
    append_run_log(ts_utc=ts, step="evaluate_predictions", status="OK" if rc==0 else f"FAIL({rc})", stdout=out[-2000:])

    # 3) Entraînements
    for t in TICKERS:
        cmd = [sys.executable, str(SCRIPTS / "train_next_close.py"),
               "--symbol", t,
               "--time-step", str(HP["TIME_STEP"]),
               "--epochs", str(HP["EPOCHS"]),
               "--fine-tune-epochs", str(HP["FINE_TUNE_EPOCHS"]),
               "--lookback-days", str(HP["LOOKBACK_DAYS"]),
               "--half-life-days", str(HP["HALF_LIFE_DAYS"]),
               "--start-date", HP["START_DATE"],
               "--step-days", str(HP["STEP_DAYS"]),
               "--test-days", str(HP["TEST_DAYS"])]
        out, rc = run(cmd, capture=True)
        append_run_log(ts_utc=ts, step="train_lstm", symbol=t, status="OK" if rc==0 else f"FAIL({rc})", stdout=out[-2000:])

    for t in TICKERS:
        cmd = [sys.executable, str(SCRIPTS / "train_tabular_baseline.py"),
               "--symbol", t,
               "--lookback-days", str(HP["LOOKBACK_DAYS"]),
               "--half-life-days", str(HP["HALF_LIFE_DAYS"])]
        out, rc = run(cmd, capture=True)
        append_run_log(ts_utc=ts, step="train_tab", symbol=t, status="OK" if rc==0 else f"FAIL({rc})", stdout=out[-2000:])

    # 4) Sweep + Backtest
    out, rc = run([sys.executable, str(SCRIPTS / "threshold_sweep.py"),
                   "--fees", str(HP["FEES"]), "--symbol", "AAPL"], capture=True)
    append_run_log(ts_utc=ts, step="threshold_sweep", status="OK" if rc==0 else f"FAIL({rc})", stdout=out[-2000:])

    out, rc = run([sys.executable, str(SCRIPTS / "evaluate_backtest.py")], capture=True)
    append_run_log(ts_utc=ts, step="evaluate_backtest", status="OK" if rc==0 else f"FAIL({rc})", stdout=out[-2000:])

    # 5) Prédictions du jour
    rows, ok = [], 0
    for t in TICKERS:
        cmd = [sys.executable, str(SCRIPTS / "predict_next_close.py"),
               "--symbol", t, "--time-step", str(HP["TIME_STEP"]), "--alpha", str(HP["ALPHA"])]
        out, rc = run(cmd, capture=True)
        m = re.search(rf"{t}\s*->\s*last_close=([\d\.]+)\s+pred_close=([\d\.]+)\s+d_pct=([-\d\.]+)", out or "")
        if m:
            last_close, pred_close, d_pct = m.groups()
            try:
                last_open = f"{get_last_open(t):.2f}"
            except Exception:
                last_open = ""
            rows.append([ts, t, last_open, last_close, pred_close, d_pct, "OK"]); ok += 1
        else:
            rows.append([ts, t, "", "", "", "", "FAIL"])
        append_run_log(ts_utc=ts, step="predict", symbol=t, status="OK" if m else "FAIL", stdout=out[-2000:])

    write_run_summary(rows)
    if ok == 0:
        print("Arrêt: toutes les prédictions ont échoué."); sys.exit(1)

    print(f"\nTerminé. Résumé: {DATA / 'run_summary.csv'}")
    print(f"Log: {RUN_LOG}")
    sys.exit(0)

if __name__ == "__main__":
    main()
