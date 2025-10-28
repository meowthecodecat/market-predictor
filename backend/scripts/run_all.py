# FILENAME: scripts/run_all.py
# -*- coding: utf-8 -*-
"""
Orchestration + auto HP + data freeze option + logging hash.
"""
import subprocess, sys, csv, re, json, hashlib
from datetime import datetime, timezone
from pathlib import Path

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

TICKERS = ["AAPL","NVDA","AMD","TSLA","AMZN","GOOGL","MSFT","KO"]
FREEZE_DATA = False  # live mode

HP = {
    "LOOKBACK_DAYS": 540,
    "HALF_LIFE_DAYS": 90,
    "START_DATE": "2019-01-01",
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

def write_run_summary(rows: list[list[str]]) -> None:
    write_header = not RUN_SUMMARY.exists()
    with RUN_SUMMARY.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header: w.writerow(["ts_utc","symbol","last_close","pred_close","d_pct","status"])
        w.writerows(rows)

def load_best_hparams() -> dict:
    hyper_dirs = [p for p in REPORTS.glob("hyper_*") if p.is_dir()]
    if not hyper_dirs: return {}
    latest = max(hyper_dirs, key=lambda p: p.name.replace("hyper_", ""))
    best = latest / "best.json"
    if not best.exists(): return {}
    try:
        with best.open("r", encoding="utf-8") as f: obj = json.load(f)
    except Exception: return {}
    out = {}
    br = obj.get("best_reg") or {}
    bc = obj.get("best_classif") or {}
    if "lookback_days" in br: out["LOOKBACK_DAYS"] = int(br["lookback_days"])
    if "half_life_days" in br: out["HALF_LIFE_DAYS"] = int(br["half_life_days"])
    if "start_date" in br: out["START_DATE"] = str(br["start_date"])
    if "time_step" in br: out["TIME_STEP"] = int(br["time_step"])
    if "epochs" in br: out["EPOCHS"] = int(br["epochs"])
    if "fine_tune_epochs" in br: out["FINE_TUNE_EPOCHS"] = int(br["fine_tune_epochs"])
    if "alpha" in br: out["ALPHA"] = float(br["alpha"])
    if "fees" in bc: out["FEES"] = float(bc["fees"])
    return out

def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def data_hash() -> str:
    files = sorted(DATA.glob("*_historical_prices.csv"))
    h = hashlib.md5()
    for fp in files: h.update(md5_file(fp).encode())
    return h.hexdigest()

def main():
    ts = now_utc()
    best = load_best_hparams(); HP.update(best)
    append_run_log(ts_utc=ts, step="load_best_hparams", status="OK" if best else "DEFAULTS", hp=json.dumps(HP))

    if not FREEZE_DATA:
        run([sys.executable, str(SCRIPTS / "data_collection.py"),
             "--symbols", ",".join(TICKERS), "--years", "10"])
        append_run_log(ts_utc=ts, step="collect", status="OK", symbols=",".join(TICKERS), data_hash=data_hash())
    else:
        append_run_log(ts_utc=ts, step="collect", status="SKIPPED", symbols=",".join(TICKERS), data_hash=data_hash())

    for t in TICKERS:
        for p in MODELS.glob(f"nextclose_{t}.*"):
            try: p.unlink()
            except Exception: pass
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
        append_run_log(ts_utc=ts, step="train_lstm", symbol=t, status="OK" if rc==0 else f"FAIL({rc})",
                       stdout=out.strip()[:2000], data_hash=data_hash())

    for t in TICKERS:
        cmd = [sys.executable, str(SCRIPTS / "train_tabular_baseline.py"),
               "--symbol", t,
               "--lookback-days", str(HP["LOOKBACK_DAYS"]),
               "--half-life-days", str(HP["HALF_LIFE_DAYS"])]
        out, rc = run(cmd, capture=True)
        append_run_log(ts_utc=ts, step="train_tab", symbol=t, status="OK" if rc==0 else f"FAIL({rc})",
                       stdout=out.strip()[:2000], data_hash=data_hash())

    out, rc = run([sys.executable, str(SCRIPTS / "threshold_sweep.py"),
                   "--fees", str(HP["FEES"]), "--symbol", "AAPL"], capture=True)
    append_run_log(ts_utc=ts, step="threshold_sweep", status="OK" if rc==0 else f"FAIL({rc})", stdout=out.strip()[:2000])

    out, rc = run([sys.executable, str(SCRIPTS / "evaluate_backtest.py")], capture=True)
    append_run_log(ts_utc=ts, step="evaluate_backtest", status="OK" if rc==0 else f"FAIL({rc})",
                   stdout=out.strip()[:2000])

    rows, ok = [], 0
    for t in TICKERS:
        cmd = [sys.executable, str(SCRIPTS / "predict_next_close.py"),
               "--symbol", t, "--time-step", str(HP["TIME_STEP"])]
        out, rc = run(cmd, capture=True)
        m = re.search(rf"{t}\s*->\s*last_close=([\d\.]+)\s+pred_close=([\d\.]+)\s+d_pct=([-\d\.]+)", out or "")
        if m:
            last_close, pred_close, d_pct = m.groups()
            rows.append([ts, t, last_close, pred_close, d_pct, "OK"]); ok += 1
        else:
            rows.append([ts, t, "", "", "", "FAIL"])
        append_run_log(ts_utc=ts, step="predict", symbol=t, status="OK" if m else "FAIL",
                       stdout=out.strip()[:2000], data_hash=data_hash())
    write_run_summary(rows)
    if ok == 0:
        print("Arrêt: toutes les prédictions ont échoué."); sys.exit(1)
    print(f"\nTerminé. Résumé: {RUN_SUMMARY}")
    print(f"Log: {RUN_LOG}")
    sys.exit(0)

if __name__ == "__main__":
    main()
