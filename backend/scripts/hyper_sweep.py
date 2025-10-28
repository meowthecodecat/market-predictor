# FILENAME: scripts/hyper_sweep.py
# -*- coding: utf-8 -*-
"""
Hyperparameter sweep élargi:
- Ajoute lookback_days, half_life_days, start_date
- Conserve structure parallèle (grid/random)
- Lance les jobs sur train_next_close.py et train_tabular_baseline.py

Sorties:
  reports/hyper_<timestamp>/classif_results.csv
  reports/hyper_<timestamp>/reg_results.csv
  reports/hyper_<timestamp>/best.json
"""

import itertools, random, re, csv, json, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count

# === chemins ===
SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND = SCRIPTS_DIR.parent
ROOT = BACKEND.parent
DATA = ROOT / "data"
OUT = BACKEND / "reports"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL","NVDA","AMD","TSLA","AMZN","GOOGL","MSFT","KO"]

CFG = {
    "mode": "grid",           # "grid" ou "random"
    "n_random": 200,
    "parallel_workers": max(1, cpu_count() // 2),
    # Hyperparams supplémentaires
    "meta": {
        "lookback_days": [360, 540, 720],
        "half_life_days": [30, 60, 90],
        "start_date": ["2018-01-01","2019-01-01"]
    },
    "classif": {
        "threshold": [0.45,0.50,0.55,0.60],
        "time_step": [20,30,40],
        "fees": [0.0001,0.0002,0.0005],
        "cooldown": [0,1,2],
        "label": ["up_1d"],
    },
    "reg": {
        "time_step": [20,30,40],
        "epochs": [10,20,30],
        "fine_tune_epochs": [0,8,12],
        "alpha": [0.4,0.5,0.6,0.7],
    }
}

TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUT / f"hyper_{TS}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CSV_CLASSIF = RUN_DIR / "classif_results.csv"
CSV_REG = RUN_DIR / "reg_results.csv"
JSON_BEST = RUN_DIR / "best.json"

# === Helpers ===
def _run_py(script: Path, args: list[str], capture: bool = True):
    cmd = [sys.executable, str(script), *map(str, args)]
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture)
    if res.returncode != 0:
        return False, (res.stdout or "") + "\n" + (res.stderr or "")
    return True, res.stdout or ""

def all_combos(space: dict):
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    for prod in itertools.product(*vals):
        yield dict(zip(keys, prod))

def random_combos(space: dict, n: int):
    keys = list(space.keys())
    pools = [space[k] for k in keys]
    for _ in range(n):
        yield {k: random.choice(pool) for k, pool in zip(keys, pools)}

# === jobs top-level ===
def eval_reg(h: dict):
    # injection des meta hyperparams dans l'appel train_next_close.py
    rmse_list, mae_list = [], []
    for t in TICKERS:
        args = [
            "--symbol", t,
            "--time-step", h["time_step"],
            "--epochs", h["epochs"],
            "--fine-tune-epochs", h["fine_tune_epochs"],
            "--lookback-days", h["lookback_days"],
            "--half-life-days", h["half_life_days"],
            "--start-date", h["start_date"],
        ]
        ok, out = _run_py(SCRIPTS_DIR / "train_next_close.py", args)
        m_rmse = re.search(r"RMSE_val_terminal\s*=\s*([0-9\.]+)", out or "", re.I)
        m_mae  = re.search(r"MAE_val_terminal\s*=\s*([0-9\.]+)", out or "", re.I)
        if m_rmse: rmse_list.append(float(m_rmse.group(1)))
        if m_mae:  mae_list.append(float(m_mae.group(1)))

    rmse = sum(rmse_list)/len(rmse_list) if rmse_list else float("nan")
    mae  = sum(mae_list)/len(mae_list) if mae_list else float("nan")

    return {**h, "rmse_val": rmse, "mae_val": mae}

def eval_classif(h: dict):
    # utilise train_tabular_baseline avec meta params
    args = [
        "--symbol", "AAPL",
        "--lookback-days", h["lookback_days"],
        "--half-life-days", h["half_life_days"],
        "--start-date", h["start_date"],
    ]
    ok, out = _run_py(SCRIPTS_DIR / "train_tabular_baseline.py", args)
    m_acc = re.search(r"acc\s*[:=]\s*([0-9\.]+)", out or "")
    m_f1 = re.search(r"f1\s*[:=]\s*([0-9\.]+)", out or "")
    m_auc = re.search(r"auc\s*[:=]\s*([0-9\.]+)", out or "")
    acc = float(m_acc.group(1)) if m_acc else float("nan")
    f1  = float(m_f1.group(1)) if m_f1 else float("nan")
    auc = float(m_auc.group(1)) if m_auc else float("nan")
    return {**h, "acc": acc, "f1": f1, "auc": auc}

def write_csv(path: Path, rows: list[dict], header: list[str]):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        for r in rows: w.writerow(r)

# === sweep ===
def sweep(section: str):
    base_space = CFG[section]
    meta_space = CFG["meta"]
    merged_space = {**base_space, **meta_space}

    gen = all_combos(merged_space) if CFG["mode"]=="grid" else random_combos(merged_space, CFG["n_random"])
    fn = eval_classif if section=="classif" else eval_reg
    header = list(merged_space.keys()) + (["acc","f1","auc"] if section=="classif" else ["rmse_val","mae_val"])
    csv_path = CSV_CLASSIF if section=="classif" else CSV_REG

    jobs = list(gen)
    results = []
    with Pool(processes=CFG["parallel_workers"]) as pool:
        for r in pool.imap_unordered(fn, jobs):
            results.append(r)
            write_csv(csv_path, [r], header)

    # tri
    if section=="classif":
        results = [r for r in results if not any(x!=x for x in [r["f1"], r["acc"]])]
        results.sort(key=lambda r: (r["f1"], r["acc"]), reverse=True)
    else:
        results = [r for r in results if not any(x!=x for x in [r["rmse_val"], r["mae_val"]])]
        results.sort(key=lambda r: (r["rmse_val"], r["mae_val"]))
    return results[0] if results else None

def main():
    print("=== Sweep Hyperparams ===")
    best_c = sweep("classif")
    best_r = sweep("reg")
    with JSON_BEST.open("w", encoding="utf-8") as f:
        json.dump({"best_classif": best_c, "best_reg": best_r}, f, ensure_ascii=False, indent=2)
    print("\n=== MEILLEURS PARAMÈTRES ===")
    print(json.dumps({"best_classif": best_c, "best_reg": best_r}, indent=2))
    print(f"\nRésultats CSV: {CSV_CLASSIF}\n             : {CSV_REG}\nRésumé JSON : {JSON_BEST}")

if __name__ == "__main__":
    main()
