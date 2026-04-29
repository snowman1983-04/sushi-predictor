"""Adapter for the SIGNATE 'お弁当の需要予測' dataset (zeroinc).

Treats the entire series as a single product (product_id='BENTO') per the
A-plan: the menu name (`name`) is intentionally not turned into features here,
so this is a pure date+weather baseline comparable to the sushi benchmark.

train.csv has the target `y`; test.csv does not (it's the held-out evaluation
set with no public labels). For a CV-based benchmark we only use train.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.feature_engineering import is_holiday

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "datasets" / "zeroinc" / "train.csv"

# 7 raw weather labels → 4 standard buckets.
# 雷電 (thunder) is treated as rainy since it implies precipitation.
_WEATHER_MAP = {
    "快晴": "sunny",
    "晴れ": "sunny",
    "薄曇": "cloudy",
    "曇": "cloudy",
    "雨": "rainy",
    "雷電": "rainy",
    "雪": "snowy",
}

_DOW_MAP = {
    "月": "Monday",
    "火": "Tuesday",
    "水": "Wednesday",
    "木": "Thursday",
    "金": "Friday",
    "土": "Saturday",
    "日": "Sunday",
}


def _parse_precipitation(value: object) -> float:
    """'--' (no rain) → 0.0; numeric strings → float."""
    if pd.isna(value):
        return 0.0
    s = str(value).strip()
    if s in {"--", ""}:
        return 0.0
    return float(s)


def load(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8")
    raw["date"] = pd.to_datetime(raw["datetime"])

    out = pd.DataFrame({
        "date": raw["date"],
        "product_id": "BENTO",
        "product_name": raw["name"].fillna("").astype(str),
        "day_of_week": raw["week"].map(_DOW_MAP),
        "weather": raw["weather"].map(_WEATHER_MAP),
        "temperature": raw["temperature"].astype(float),
        "precipitation": raw["precipitation"].apply(_parse_precipitation),
        "sales_count": raw["y"].astype(int),
    })

    if out["day_of_week"].isna().any():
        bad = raw.loc[out["day_of_week"].isna(), "week"].unique()
        raise ValueError(f"unmapped day_of_week values: {bad}")
    if out["weather"].isna().any():
        bad = raw.loc[out["weather"].isna(), "weather"].unique()
        raise ValueError(f"unmapped weather values: {bad}")

    dow_idx = out["date"].dt.dayofweek
    out["is_weekend"] = dow_idx >= 5
    out["is_holiday"] = out["date"].dt.date.apply(is_holiday)
    # `payday` flag in the raw file marks pay-day (給料日, 末日寄り) — semantically
    # closest to is_pension_day in the sushi schema (年金支給日 / 給料日系の所得イベント).
    out["is_pension_day"] = raw["payday"].fillna(0).astype(int).astype(bool)
    # No equivalent of サービスデー in the bento data.
    out["is_sale_day"] = False

    # No price / CPI data — fill with neutral constants so downstream features
    # (which expect these columns) get a no-op value.
    out["effective_price"] = 0.0
    out["cpi_index"] = 100.0

    return out
