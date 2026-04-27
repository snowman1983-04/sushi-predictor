"""Generate dummy sushi sales data per spec.md section 4.

Output: data/sushi_sales.csv
Period: 2024-04-01 to 2026-03-31 (730 days) x 4 products = 2920 rows.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE = date(2024, 4, 1)
END_DATE = date(2026, 3, 31)

PRODUCTS = [
    {"product_id": "P001", "product_name": "水木寿司（10貫）", "base": 12, "sale_only": False},
    {"product_id": "P002", "product_name": "8貫寿司",          "base": 10, "sale_only": False},
    {"product_id": "P003", "product_name": "10貫寿司",         "base": 15, "sale_only": False},
    {"product_id": "P004", "product_name": "ランチ寿司",       "base": 20, "sale_only": True},
]

# Hirosaki monthly mean temperature (deg C), index 1..12
MONTHLY_MEAN_TEMP = {
    1: -2.0, 2: -1.0, 3: 3.0, 4: 9.0, 5: 15.0, 6: 19.0,
    7: 23.0, 8: 25.0, 9: 20.0, 10: 13.0, 11: 7.0, 12: 1.0,
}

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the date of the n-th occurrence of `weekday` (Mon=0..Sun=6) in (year, month)."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def japanese_holidays(year: int) -> set[date]:
    """Approximate set of Japanese national holidays for the given year.

    Substitute holidays are not implemented (close enough for synthetic data).
    """
    holidays: set[date] = set()
    holidays.add(date(year, 1, 1))                      # 元日
    holidays.add(nth_weekday_of_month(year, 1, 0, 2))   # 成人の日 (Jan 2nd Monday)
    holidays.add(date(year, 2, 11))                     # 建国記念の日
    holidays.add(date(year, 2, 23))                     # 天皇誕生日
    holidays.add(date(year, 3, 20))                     # 春分の日 (近似)
    holidays.add(date(year, 4, 29))                     # 昭和の日
    holidays.add(date(year, 5, 3))                      # 憲法記念日
    holidays.add(date(year, 5, 4))                      # みどりの日
    holidays.add(date(year, 5, 5))                      # こどもの日
    holidays.add(nth_weekday_of_month(year, 7, 0, 3))   # 海の日 (Jul 3rd Monday)
    holidays.add(date(year, 8, 11))                     # 山の日
    holidays.add(nth_weekday_of_month(year, 9, 0, 3))   # 敬老の日 (Sep 3rd Monday)
    holidays.add(date(year, 9, 23))                     # 秋分の日 (近似)
    holidays.add(nth_weekday_of_month(year, 10, 0, 2))  # スポーツの日 (Oct 2nd Monday)
    holidays.add(date(year, 11, 3))                     # 文化の日
    holidays.add(date(year, 11, 23))                    # 勤労感謝の日
    return holidays


def is_pension_day(d: date) -> bool:
    """Pension payday: the 15th of even months."""
    return d.month % 2 == 0 and d.day == 15


def sample_weather(month: int, rng: np.random.Generator) -> str:
    """Sample weather. Winter months (Dec-Mar) get more snow, fewer sunny days."""
    if month in (12, 1, 2, 3):
        probs = [0.30, 0.30, 0.15, 0.25]  # sunny / cloudy / rainy / snowy
    elif month in (6, 7):
        probs = [0.35, 0.30, 0.35, 0.0]   # tsuyu (rainy season)
    else:
        probs = [0.55, 0.25, 0.18, 0.02]
    return rng.choice(["sunny", "cloudy", "rainy", "snowy"], p=probs)


def sample_temperature(d: date, rng: np.random.Generator) -> float:
    """Daily mean temperature: monthly mean + N(0, 3)."""
    mean = MONTHLY_MEAN_TEMP[d.month]
    return float(np.round(mean + rng.normal(0, 3.0), 1))


def sample_precipitation(weather: str, rng: np.random.Generator) -> float:
    if weather == "sunny":
        return 0.0
    if weather == "cloudy":
        return float(np.round(rng.uniform(0.0, 1.0), 1))
    if weather == "rainy":
        return float(np.round(rng.uniform(3.0, 15.0), 1))
    if weather == "snowy":
        return float(np.round(rng.uniform(1.0, 10.0), 1))
    raise ValueError(f"Unknown weather: {weather}")


def compute_sales(
    base: int,
    sale_only: bool,
    is_sale: bool,
    is_wed: bool,
    is_weekend: bool,
    is_pension: bool,
    is_holiday: bool,
    weather: str,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if sale_only and not is_sale:
        return 0

    multiplier = 1.0
    if is_wed and is_sale:
        multiplier *= 2.0
    if is_weekend:
        multiplier *= 1.5
    if is_pension:
        multiplier *= 1.8
    if is_holiday:
        multiplier *= 1.4
    if weather == "rainy":
        multiplier *= 0.7
    elif weather == "snowy":
        multiplier *= 0.5
    if temperature >= 30:
        multiplier *= 0.8
    elif temperature <= 0:
        multiplier *= 0.9

    expected = base * multiplier
    noise = rng.normal(0, expected * 0.15)
    return int(max(0, round(expected + noise)))


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # Pre-compute holidays for every year touched by the date range.
    holiday_set: set[date] = set()
    for year in range(START_DATE.year, END_DATE.year + 1):
        holiday_set |= japanese_holidays(year)

    rows: list[dict] = []
    current = START_DATE
    while current <= END_DATE:
        dow_idx = current.weekday()  # Mon=0..Sun=6
        is_wed = dow_idx == 2
        is_weekend = dow_idx >= 5
        is_holiday = current in holiday_set
        is_pension = is_pension_day(current)
        is_sale = is_wed or is_weekend

        weather = sample_weather(current.month, rng)
        temperature = sample_temperature(current, rng)
        precipitation = sample_precipitation(weather, rng)

        for product in PRODUCTS:
            sales = compute_sales(
                base=product["base"],
                sale_only=product["sale_only"],
                is_sale=is_sale,
                is_wed=is_wed,
                is_weekend=is_weekend,
                is_pension=is_pension,
                is_holiday=is_holiday,
                weather=weather,
                temperature=temperature,
                rng=rng,
            )
            rows.append({
                "date": current.isoformat(),
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "day_of_week": DOW_NAMES[dow_idx],
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "is_pension_day": is_pension,
                "is_sale_day": is_sale,
                "weather": weather,
                "temperature": temperature,
                "precipitation": precipitation,
                "sales_count": sales,
            })

        current += timedelta(days=1)

    return pd.DataFrame(rows)


def main() -> None:
    df = generate()
    out_path = Path(__file__).parent / "sushi_sales.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
    print("\nSummary stats (sales_count):")
    print(df.groupby("product_id")["sales_count"].describe())


if __name__ == "__main__":
    main()
