# backend/server/services/confidence.py
import numpy as np

def confidence_from_residual_sigma(residual: float, sigma: float, clip: float = 3.0) -> float:
    # map |residual| vs dispersion -> score 0..1
    if sigma <= 1e-12:
        return 1.0
    z = min(abs(residual) / sigma, clip)
    return float(max(0.0, 1.0 - z/clip))

def batch_confidence(preds, actuals_window):
    # preds: list of (pred_close, last_close)
    # actuals_window: np.array of historical abs residuals -> compute sigma
    sigma = float(np.std(actuals_window)) if len(actuals_window) else 0.0
    out = []
    for pred_close, last_close in preds:
        residual = pred_close - last_close
        out.append(confidence_from_residual_sigma(residual, sigma))
    return out
