"""Load the sushi sales CSV and return a typed DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sushi_sales.csv"


def load_data(path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "product_id"]).reset_index(drop=True)
    return df
