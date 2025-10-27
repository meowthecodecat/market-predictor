# -*- coding: utf-8 -*-
import itertools, random, re, csv, json, os, subprocess
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL","NVDA","AMD","TSLA","AMZN","GOOGL","MSFT","KO"]

# ==== CONFIG ====
CFG = {
    "mode": "grid",                 # "grid" ou "random"
    "n_random": 200,                # utile si mode=random
    "parallel_workers": max(1, cpu_count()//2),

    # Espace hyperparamètres
    "classif": {
        "threshold": [0.45,0.50,0.55,0.60],
        "time_step": [20,30,40],
        "fees": [0.0001, 0.0002, 0.0005],
        "cooldown": [0,1,2],
        "label": ["up_1d"],
    },
    "reg": {
        "time_step": [20,30,40],
        "epochs": [10,20,30],
        "fine_tune_epochs": [0,8,12],
        "alpha": [0.4,0.5,0.6,0.7],  # mélange LSTM/GBM pour la prédiction finale
    }
}

TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUT / f"hyper_{TS}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CSV_CLASSIF = RUN_DIR / "classif_results.csv"
CSV_REG = RUN_DIR / "reg_results.csv"
JSON_BEST = RUN_DIR / "best.json"

def run(cmd, capture=True):
    res = subprocess.run(cmd, shell=True, text=True, capture_output=capture, cwd=ROOT)
    if res.returncode != 0:
        return False, res.stdout + "\n" + res.stderr
    return True, res.stdout

def all_combos(d):
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    for prod in itertools.product(*vals):
        yield dict(zip(keys, prod))

def random_combos(d, n):
    keys = list(d.keys())
    pools = [d[k] for k in keys]
    for _ in range(n):
        yield {k: random.choice(pool) for k, pool in zip(keys, pools)}

# ---------- Classification sweep ----------
def eval_classif(h):
    # 1) backtest global sur label (utilise evaluate_backtest.py)
    cmd = (
        f"python scripts/evaluate_backtest.py "
        f"--threshold {h['threshold']} --label {h['label']} "
        f"--fees {h['fees']} --cooldown {h['cooldown']}"
    )
    ok, out = run(cmd)
    # Parsers robustes
    m_perf = re.search(r"Perf cumulée\s*:\s*([-\d\.]+)%", out)
    m_sharpe = re.search(r"Sharpe approx\s*:\s*([-\d\.]+)", out)
    m_mdd = re.search(r"Max drawdown\s*:\s*([-\d\.]+)%", out)
    perf = float(m_perf.group(1)) if m_perf else float("nan")
    sharpe = float(m_sharpe.group(1)) if m_sharpe else float("nan")
    mdd = float(m_mdd.group(1)) if m_mdd else float("nan")
    row = {
        **h, "perf_pct": perf, "sharpe": sharpe, "mdd_pct": mdd,
        "stdout": out.strip().replace("\r"," ")
    }
    return row

# ---------- Régression sweep ----------
def eval_reg(h):
    # 1) entraîner par ticker puis moyenner métriques de validation
    rmse_list, mae_list = [], []
    logs = []
    for t in TICKERS:
        cmd = (
            f"python scripts/train_next_close.py "
            f"--symbol {t} --time-step {h['time_step']} "
            f"--epochs {h['epochs']} --fine-tune-epochs {h['fine_tune_epochs']}"
        )
        ok, out = run(cmd)
        # train_next_close.py doit imprimer: "RMSE_val=..., MAE_val=..."
        m_rmse = re.search(r"RMSE[_\s]?val\s*=\s*([0-9\.]+)", out, re.I)
        m_mae  = re.search(r"MAE[_\s]?val\s*=\s*([0-9\.]+)", out, re.I)
        if m_rmse: rmse_list.append(float(m_rmse.group(1)))
        if m_mae: mae_list.append(float(m_mae.group(1)))
        logs.append({ "symbol": t, "log": out.strip().replace("\r"," ") })
    rmse = sum(rmse_list)/len(rmse_list) if rmse_list else float("nan")
    mae  = sum(mae_list)/len(mae_list) if mae_list else float("nan")

    # 2) prédiction next close multi-titres pour vérifier stabilité du pipeline
    pred_ok, pred_out = run(
        f"python scripts/ensemble_predict.py --symbols {','.join(TICKERS)} "
        f"--threshold 0.50 --label up_1d --time-step {h['time_step']} --alpha {h['alpha']}"
    )
    row = { **h, "rmse_val": rmse, "mae_val": mae, "stdout": pred_out.strip().replace("\r"," ") }
    return row

def write_csv(path, rows, header):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        for r in rows: w.writerow(r)

def sweep(section):
    space = CFG[section]
    gen = all_combos(space) if CFG["mode"]=="grid" else random_combos(space, CFG["n_random"])
    fn = eval_classif if section=="classif" else eval_reg
    header = (
        list(space.keys()) +
        (["perf_pct","sharpe","mdd_pct","stdout"] if section=="classif"
         else ["rmse_val","mae_val","stdout"])
    )
    csv_path = CSV_CLASSIF if section=="classif" else CSV_REG

    # parallélisation
    jobs = list(gen)
    chunk = []
    results = []
    def do(h): return fn(h)
    with Pool(processes=CFG["parallel_workers"]) as pool:
        for r in pool.imap_unordered(do, jobs):
            results.append(r)
            chunk.append(r)
            if len(chunk)>=10:
                write_csv(csv_path, chunk, header); chunk=[]
        if chunk:
            write_csv(csv_path, chunk, header)

    # sélection du meilleur
    best = None
    if section=="classif":
        # critère: Sharpe max, puis perf_pct
        results = [r for r in results if not any(map(lambda x: x!=x,[r["sharpe"],r["perf_pct"]]))]  # remove NaN
        results.sort(key=lambda r: (r["sharpe"], r["perf_pct"]), reverse=True)
        best = results[0] if results else None
    else:
        # critère: RMSE min, puis MAE min
        results = [r for r in results if not any(map(lambda x: x!=x,[r["rmse_val"],r["mae_val"]]))]
        results.sort(key=lambda r: (r["rmse_val"], r["mae_val"]))
        best = results[0] if results else None
    return best

def main():
    # 0) collecte des données à jour
    run(f"python scripts/data_collection.py --symbols {','.join(TICKERS)} --years 10", capture=False)

    best_c = sweep("classif")
    best_r = sweep("reg")

    with JSON_BEST.open("w", encoding="utf-8") as f:
        json.dump({"best_classif": best_c, "best_reg": best_r}, f, ensure_ascii=False, indent=2)

    print("\n=== MEILLEURS PARAMÈTRES ===")
    print(json.dumps({"best_classif": best_c, "best_reg": best_r}, ensure_ascii=False, indent=2))
    print(f"\nRésultats CSV: {CSV_CLASSIF}\n             : {CSV_REG}\nRésumé JSON : {JSON_BEST}")

if __name__ == "__main__":
    main()
