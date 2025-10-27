# -*- coding: utf-8 -*-
import logging, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

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
        logger.addHandler(fh); logger.addHandler(sh)
    return logger
