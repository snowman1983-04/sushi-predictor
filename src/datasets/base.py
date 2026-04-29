"""Shared schema and dispatcher for dataset adapters.

All adapters MUST return a DataFrame with the columns in ``STANDARD_COLUMNS``,
sorted by (product_id, date). Downstream code (feature_engineering, trainer)
relies on this exact set.
"""

from __future__ import annotations

import pandas as pd

STANDARD_COLUMNS = [
    "date",            # datetime64[ns]
    "product_id",      # str (series identifier)
    "product_name",    # str
    "day_of_week",     # str: Monday..Sunday
    "is_weekend",      # bool
    "is_holiday",      # bool
    "is_pension_day",  # bool
    "is_sale_day",     # bool
    "weather",         # str: sunny/cloudy/rainy/snowy
    "temperature",     # float
    "precipitation",   # float (>=0)
    "effective_price", # float
    "cpi_index",       # float
    "sales_count",     # int
]


def load_dataset(name: str) -> pd.DataFrame:
    """Dispatch to the named adapter and return a normalized DataFrame."""
    if name == "sushi":
        from src.datasets.sushi import load as _load
    elif name == "bento":
        from src.datasets.bento import load as _load
    else:
        raise ValueError(f"Unknown dataset {name!r}. Known: sushi, bento")

    df = _load()
    missing = set(STANDARD_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"adapter '{name}' missing columns: {sorted(missing)}")
    return df[STANDARD_COLUMNS].sort_values(["product_id", "date"]).reset_index(drop=True)
