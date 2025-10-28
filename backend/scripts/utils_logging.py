# FILENAME: scripts/utils_logging.py
# -*- coding: utf-8 -*-
"""
Logger + journal CSV des runs.
- init_logger(name, log_dir, level)
- append_run_csv(csv_path, **fields)
"""

import logging, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

def init_logger(name="market", log_dir="logs", level="INFO"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    fh = RotatingFileHandler(Path(log_dir) / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def append_run_csv(csv_path: Path, **fields: Any) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not csv_path.exists()
    # écriture simple sans pandas
    import csv
    keys = list(fields.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if header_needed:
            w.writeheader()
        w.writerow(fields)
