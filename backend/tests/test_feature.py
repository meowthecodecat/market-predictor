import pandas as pd
from scripts.data_preprocessing import make_features_from_df  # ajuste si nom différent

def test_make_features_basic():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=60, freq="B"),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000
    })
    out, cols = make_features_from_df(df)
    assert len(out) == 60
    assert len(cols) > 0
