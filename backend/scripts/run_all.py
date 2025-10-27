# -*- coding: utf-8 -*-
"""
Pipeline complet robuste:
1) collecte
2) train LSTM par ticker
3) prédiction next close par ticker
4) écrit data/run_summary.csv
Continue si un ticker échoue, s'arrête si tous échouent à l'étape prédiction.
"""
import subprocess, sys, csv, re
from datetime import datetime
from pathlib import Path

TICKERS = ["AAPL","NVDA","AMD","TSLA","AMZN","GOOGL","MSFT","KO"]
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = DATA / "run_summary.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

def run(cmd, capture=False):
    print(f"\n>>> {cmd}\n")
    try:
        res = subprocess.run(cmd, shell=True, check=True, capture_output=capture, text=True)
        return (res.stdout if capture else ""), 0
    except subprocess.CalledProcessError as e:
        if capture:
            return e.stdout + "\n" + e.stderr, e.returncode
        return "", e.returncode

def append_rows(rows):
    write_header = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ts_utc","symbol","last_close","pred_close","d_pct","status"])
        w.writerows(rows)

def main():
    # 1) Collecte
    cmd_collect = f'python "{SCRIPTS / "data_collection.py"}" --symbols {",".join(TICKERS)} --years 10'
    _, _ = run(cmd_collect, capture=False)

    # 2) Train
    for t in TICKERS:
        # purge artefacts existants pour éviter désalignements
        for p in MODELS.glob(f"nextclose_{t}.*"):
            try:
                p.unlink()
            except Exception:
                pass
        cmd_train = f'python "{SCRIPTS / "train_next_close.py"}" --symbol {t} --time-step 30 --epochs 20 --fine-tune-epochs 12'
        _, _ = run(cmd_train, capture=False)

    # 3) Predict
    rows = []
    ok = 0
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for t in TICKERS:
        out, rc = run(f'python "{SCRIPTS / "predict_next_close.py"}" --symbol {t} --time-step 30 --alpha 0.6', capture=True)
        m = re.search(rf"{t}\s*->\s*last_close=([\d\.]+)\s+pred_close=([\d\.]+)\s+d_pct=([-\d\.]+)", out)
        if m:
            last_close, pred_close, d_pct = m.groups()
            rows.append([now, t, last_close, pred_close, d_pct, "OK"])
            ok += 1
        else:
            rows.append([now, t, "", "", "", "FAIL"])
    append_rows(rows)

    if ok == 0:
        print("Arrêt: toutes les prédictions de close ont échoué.")
        sys.exit(1)

    print(f"\nTerminé. Récap: {OUT}")
    sys.exit(0)

if __name__ == "__main__":
    main()
