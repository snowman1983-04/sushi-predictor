"""Feature engineering: derive model inputs from raw rows or user input.

Holiday and sale-day logic is re-implemented here (mirrors data/generate_data.py)
to keep src/ structurally independent of data/.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

WEATHER_CATEGORIES = ["sunny", "cloudy", "rainy", "snowy"]
DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PRODUCT_IDS = ["P001", "P002", "P003", "P004"]

NUMERIC_COLS = ["temperature", "precipitation", "month"]
FLAG_COLS = ["is_weekend", "is_holiday", "is_pension_day", "is_sale_day"]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def japanese_holidays(year: int) -> set[date]:
    h: set[date] = set()
    h.add(date(year, 1, 1))
    h.add(_nth_weekday(year, 1, 0, 2))
    h.add(date(year, 2, 11))
    h.add(date(year, 2, 23))
    h.add(date(year, 3, 20))
    h.add(date(year, 4, 29))
    h.add(date(year, 5, 3))
    h.add(date(year, 5, 4))
    h.add(date(year, 5, 5))
    h.add(_nth_weekday(year, 7, 0, 3))
    h.add(date(year, 8, 11))
    h.add(_nth_weekday(year, 9, 0, 3))
    h.add(date(year, 9, 23))
    h.add(_nth_weekday(year, 10, 0, 2))
    h.add(date(year, 11, 3))
    h.add(date(year, 11, 23))
    return h


def is_holiday(d: date) -> bool:
    return d in japanese_holidays(d.year)


def is_pension_day(d: date) -> bool:
    return d.month % 2 == 0 and d.day == 15


def derive_date_flags(d: date) -> dict[str, object]:
    dow_idx = d.weekday()
    is_weekend = dow_idx >= 5
    is_wed = dow_idx == 2
    return {
        "month": d.month,
        "day_of_week": DOW_NAMES[dow_idx],
        "is_weekend": is_weekend,
        "is_holiday": is_holiday(d),
        "is_pension_day": is_pension_day(d),
        "is_sale_day": is_wed or is_weekend,
    }


def _one_hot(df: pd.DataFrame, col: str, categories: list[str], prefix: str) -> pd.DataFrame:
    cat = pd.Categorical(df[col], categories=categories)
    dummies = pd.get_dummies(cat, prefix=prefix, dtype=int)
    return dummies


def _assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["temperature"] = df["temperature"].astype(float)
    out["precipitation"] = df["precipitation"].astype(float)
    out["month"] = df["month"].astype(int)
    for col in FLAG_COLS:
        out[col] = df[col].astype(int)
    out = pd.concat(
        [
            out,
            _one_hot(df, "weather", WEATHER_CATEGORIES, "weather"),
            _one_hot(df, "day_of_week", DOW_NAMES, "dow"),
        ],
        axis=1,
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix from the raw dataframe.

    Expects a ``date`` column of dtype datetime64. Adds ``month`` then assembles.
    """
    work = df.copy()
    work["month"] = work["date"].dt.month
    return _assemble_features(work)


def prepare_input(
    target_date: date | datetime | str,
    weather: str,
    temperature: float,
    precipitation: float,
) -> pd.DataFrame:
    """Build a single-row feature matrix for inference."""
    if isinstance(target_date, str):
        target_date = datetime.fromisoformat(target_date).date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()

    if weather not in WEATHER_CATEGORIES:
        raise ValueError(f"weather must be one of {WEATHER_CATEGORIES}, got {weather!r}")

    flags = derive_date_flags(target_date)
    row = {
        "temperature": temperature,
        "precipitation": precipitation,
        "weather": weather,
        **flags,
    }
    df = pd.DataFrame([row])
    return _assemble_features(df)


def feature_columns() -> list[str]:
    """Canonical column order. Useful for sanity checks."""
    cols = ["temperature", "precipitation", "month"] + FLAG_COLS
    cols += [f"weather_{w}" for w in WEATHER_CATEGORIES]
    cols += [f"dow_{d}" for d in DOW_NAMES]
    return cols


CLASS_LABELS = {0: "少", 1: "普", 2: "多"}


def fit_sales_thresholds(sales: pd.Series, low_q: float = 1 / 3, high_q: float = 2 / 3) -> tuple[float, float]:
    """Return (low, high) cut points for the 3-class discretization.

    A degenerate case (low == high) can happen when the distribution is heavily
    zero-inflated (e.g. P004). We nudge ``high`` up so that at least the top
    bucket is non-empty and ordering of bins is preserved.
    """
    low = float(sales.quantile(low_q))
    high = float(sales.quantile(high_q))
    if high <= low:
        positives = sales[sales > low]
        if len(positives) > 0:
            high = float(positives.min())
        else:
            high = low + 1.0
    return low, high


def discretize_sales(sales: pd.Series, low: float, high: float) -> pd.Series:
    """Assign each value to {0=少, 1=普, 2=多} given pre-fit thresholds."""
    labels = pd.Series(1, index=sales.index, dtype=int)
    labels[sales <= low] = 0
    labels[sales > high] = 2
    return labels
