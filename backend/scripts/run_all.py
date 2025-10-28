# FILENAME: scripts/run_all.py
# -*- coding: utf-8 -*-
"""
Pipeline complet avec hyperparamètres temporels:
1) Collecte
2) Train LSTM (fenêtre + demi-vie + walk-forward)
3) Train Tabulaire (mêmes paramètres)
4) Sweep des seuils (régimes + frais)
5) Évaluation backtest
6) Prédiction et résumé CSV

Robuste: continue si un ticker échoue; logue les runs.
"""
import subprocess, sys, csv, re
from datetime import datetime, timezone
from pathlib import Path

# ---------- Config ----------
TICKERS = ["AAPL","NVDA","AMD","TSLA","AMZN","GOOGL","MSFT","KO"]

# Hyperparams temps
LOOKBACK_DAYS = 540
HALF_LIFE_DAYS = 90
START_DATE = "2019-01-01"
STEP_DAYS = 30
TEST_DAYS = 30

# Modèle LSTM
TIME_STEP = 30
EPOCHS = 20
FINE_TUNE_EPOCHS = 12

# Décision / coûts
ALPHA = 0.60         # mix éventuel côté predict_next_close.py
FEES = 0.0002        # 2 bps
LONG_ONLY = True

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "backend" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

RUN_SUMMARY = DATA / "run_summary.csv"
RUN_LOG = REPORTS / "run_log.csv"

# ---------- Utils ----------
def run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> tuple[str, int]:
    try:
        res = subprocess.run(cmd, cwd=cwd or ROOT, text=True, capture_output=capture, check=True)
        return (res.stdout if capture else ""), 0
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "")
        return (out if capture else ""), e.returncode

def write_run_summary(rows: list[list[str]]) -> None:
    write_header = not RUN_SUMMARY.exists()
    with RUN_SUMMARY.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ts_utc","symbol","last_close","pred_close","d_pct","status"])
        w.writerows(rows)

def append_run_log(**fields) -> None:
    header_needed = not RUN_LOG.exists()
    keys = list(fields.keys())
    with RUN_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if header_needed:
            w.writeheader()
        w.writerow(fields)

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ---------- Main ----------
def main():
    ts = now_utc()

    # 1) Collecte
    out, rc = run([sys.executable, str(SCRIPTS / "data_collection.py"),
                   "--symbols", ",".join(TICKERS), "--years", "10"], capture=False)

    append_run_log(
        ts_utc=ts, step="collect", status="OK" if rc==0 else f"FAIL({rc})",
        symbols=",".join(TICKERS)
    )

    # 2) Train LSTM par ticker (purge artefacts)
    for t in TICKERS:
        for p in MODELS.glob(f"nextclose_{t}.*"):
            try: p.unlink()
            except Exception: pass

        cmd = [sys.executable, str(SCRIPTS / "train_next_close.py"),
               "--symbol", t,
               "--time-step", str(TIME_STEP),
               "--epochs", str(EPOCHS),
               "--fine-tune-epochs", str(FINE_TUNE_EPOCHS),
               "--lookback-days", str(LOOKBACK_DAYS),
               "--half-life-days", str(HALF_LIFE_DAYS),
               "--start-date", START_DATE,
               "--step-days", str(STEP_DAYS),
               "--test-days", str(TEST_DAYS)]
        out, rc = run(cmd, capture=True)
        append_run_log(
            ts_utc=ts, step="train_lstm", symbol=t, status="OK" if rc==0 else f"FAIL({rc})",
            lookback_days=LOOKBACK_DAYS, half_life_days=HALF_LIFE_DAYS,
            start_date=START_DATE, step_days=STEP_DAYS, test_days=TEST_DAYS,
            time_step=TIME_STEP, epochs=EPOCHS, fine_tune=FINE_TUNE_EPOCHS,
            stdout=out.strip()[:2000]
        )

    # 3) Train Tabulaire (mêmes hyperparams)
    for t in TICKERS:
        cmd = [sys.executable, str(SCRIPTS / "train_tabular_baseline.py"),
               "--symbol", t,
               "--lookback-days", str(LOOKBACK_DAYS),
               "--half-life-days", str(HALF_LIFE_DAYS),
               "--start-date", START_DATE,
               "--step-days", str(STEP_DAYS),
               "--test-days", str(TEST_DAYS)]
        out, rc = run(cmd, capture=True)
        append_run_log(
            ts_utc=ts, step="train_tab", symbol=t, status="OK" if rc==0 else f"FAIL({rc})",
            lookback_days=LOOKBACK_DAYS, half_life_days=HALF_LIFE_DAYS,
            start_date=START_DATE, step_days=STEP_DAYS, test_days=TEST_DAYS,
            stdout=out.strip()[:2000]
        )

    # 4) Sweep des seuils (régimes + frais)
    thr_args = [sys.executable, str(SCRIPTS / "threshold_sweep.py"),
                "--label", "up_1d", "--fees", str(FEES)]
    if LONG_ONLY: thr_args.append("--long-only")
    out, rc = run(thr_args, capture=True)
    append_run_log(ts_utc=ts, step="threshold_sweep", status="OK" if rc==0 else f"FAIL({rc})",
                   fees=FEES, stdout=out.strip()[:2000])

    # 5) Évaluation backtest (informative)
    out, rc = run([sys.executable, str(SCRIPTS / "evaluate_backtest.py")], capture=True)
    append_run_log(ts_utc=ts, step="evaluate_backtest", status="OK" if rc==0 else f"FAIL({rc})",
                   stdout=out.strip()[:2000])

    # 6) Prédictions next close
    rows, ok = [], 0
    for t in TICKERS:
        cmd = [sys.executable, str(SCRIPTS / "predict_next_close.py"),
               "--symbol", t, "--time-step", str(TIME_STEP), "--alpha", str(ALPHA)]
        out, rc = run(cmd, capture=True)
        m = re.search(rf"{t}\s*->\s*last_close=([\d\.]+)\s+pred_close=([\d\.]+)\s+d_pct=([-\d\.]+)", out or "")
        if m:
            last_close, pred_close, d_pct = m.groups()
            rows.append([ts, t, last_close, pred_close, d_pct, "OK"])
            ok += 1
        else:
            rows.append([ts, t, "", "", "", "FAIL"])
        append_run_log(ts_utc=ts, step="predict", symbol=t,
                       status="OK" if m else "FAIL", stdout=out.strip()[:2000])

    write_run_summary(rows)

    if ok == 0:
        print("Arrêt: toutes les prédictions ont échoué.")
        sys.exit(1)

    print(f"\nTerminé. Résumé: {RUN_SUMMARY}")
    print(f"Log: {RUN_LOG}")
    sys.exit(0)

if __name__ == "__main__":
    main()
