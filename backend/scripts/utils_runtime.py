# -*- coding: utf-8 -*-
import os, random, sys
import numpy as np

def set_global_seed(seed: int = 42):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)

def die(msg: str, code: int = 1):
    print(msg)
    sys.exit(code)

def validate_tickers(tickers):
    bad = [t for t in tickers if not t.isalnum()]
    if bad:
        die(f"Tickers invalides: {bad}", 2)
    return [t.upper() for t in tickers]
