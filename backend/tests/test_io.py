import json
from pathlib import Path

def test_latest_pointer(tmp_path):
    base = "nextclose_XXX_20250101_000000"
    (tmp_path/f"{base}.keras").write_text("x")
    (tmp_path/f"{base}_scaler.pkl").write_text("y")
    (tmp_path/"nextclose_XXX_LATEST.txt").write_text(base)
    assert (tmp_path/f"{base}.keras").exists()

def test_features_manifest(tmp_path):
    d = {"feature_names":["a","b"], "time_step":30}
    p = tmp_path/"f.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    j = json.loads(p.read_text(encoding="utf-8"))
    assert j["time_step"] == 30
