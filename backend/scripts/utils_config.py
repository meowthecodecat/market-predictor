# -*- coding: utf-8 -*-
from pathlib import Path
import yaml

def load_cfg(path: str | None = None):
    p = Path(path or Path(__file__).resolve().parents[1] / "config.yaml")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
