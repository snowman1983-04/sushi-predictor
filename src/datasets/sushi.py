"""Adapter for the in-house sushi_sales.csv (already in standard schema)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "sushi_sales.csv"


def load(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    return df
