"""Generate dummy sushi sales data per spec.md section 4.

Output: data/sushi_sales.csv
Period: 2024-04-01 to 2027-03-31 (1095 days) x 4 products = 4380 rows.

Updated post-Eliza review:
- Extended from 2 years to 3 years.
- Replaced the prior "sale_only" flag with explicit per-product availability
  schedules (available_dows) reflecting the actual operational pattern:
    P001 水木寿司：水・木のみ
    P002 春8貫寿司：毎日
    P003 絆10貫寿司：毎日
    P004 ランチ寿司：水・木以外
- Added effective_price (nominal, with sale-day discount) and cpi_index
  (monthly CPI multiplier) columns. Sales react to CPI via a small
  inflation drag.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE = date(2024, 4, 1)
END_DATE = date(2027, 3, 31)

# product master
# - available_dows: weekdays (Mon=0..Sun=6) on which the product is offered.
#   Outside these days the row still appears with sales=0 to keep the grid.
# - price_normal / price_sale: yen per item; price_sale applies on is_sale_day.
PRODUCTS = [
    {"product_id": "P001", "product_name": "水木寿司（10貫）", "base": 25,
     "available_dows": (2, 3),                  # Wed, Thu only
     "price_normal": 500, "price_sale": 500},   # always lowest price (集客商品)
    {"product_id": "P002", "product_name": "春8貫寿司",        "base": 10,
     "available_dows": (0, 1, 2, 3, 4, 5, 6),   # everyday
     "price_normal": 600, "price_sale": 500},
    {"product_id": "P003", "product_name": "絆10貫寿司",       "base": 12,
     "available_dows": (0, 1, 2, 3, 4, 5, 6),   # everyday
     "price_normal": 600, "price_sale": 500},
    {"product_id": "P004", "product_name": "ランチ寿司",       "base": 15,
     "available_dows": (0, 1, 4, 5, 6),         # Mon, Tue, Fri, Sat, Sun
     "price_normal": 500, "price_sale": 500},
]

# Hirosaki monthly mean temperature (deg C), index 1..12
MONTHLY_MEAN_TEMP = {
    1: -2.0, 2: -1.0, 3: 3.0, 4: 9.0, 5: 15.0, 6: 19.0,
    7: 23.0, 8: 25.0, 9: 20.0, 10: 13.0, 11: 7.0, 12: 1.0,
}

# Monthly food CPI rebased to 2024-04 = 100.0.
# Reference: 総務省統計局 消費者物価指数（食料、2020年=100基準）の推移を参考に近似。
# 出典：https://www.stat.go.jp/data/cpi/  (food category, monthly).
MONTHLY_CPI: dict[tuple[int, int], float] = {
    (2024, 4): 100.0, (2024, 5): 100.3, (2024, 6): 100.5,
    (2024, 7): 100.7, (2024, 8): 100.9, (2024, 9): 101.1,
    (2024, 10): 101.4, (2024, 11): 101.6, (2024, 12): 101.8,
    (2025, 1): 102.0, (2025, 2): 102.3, (2025, 3): 102.5,
    (2025, 4): 102.7, (2025, 5): 102.9, (2025, 6): 103.2,
    (2025, 7): 103.4, (2025, 8): 103.6, (2025, 9): 103.9,
    (2025, 10): 104.1, (2025, 11): 104.4, (2025, 12): 104.6,
    (2026, 1): 104.8, (2026, 2): 105.1, (2026, 3): 105.3,
    (2026, 4): 105.6, (2026, 5): 105.8, (2026, 6): 106.1,
    (2026, 7): 106.3, (2026, 8): 106.6, (2026, 9): 106.8,
    (2026, 10): 107.1, (2026, 11): 107.3, (2026, 12): 107.6,
    (2027, 1): 107.8, (2027, 2): 108.1, (2027, 3): 108.3,
}

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def japanese_holidays(year: int) -> set[date]:
    holidays: set[date] = set()
    holidays.add(date(year, 1, 1))
    holidays.add(nth_weekday_of_month(year, 1, 0, 2))
    holidays.add(date(year, 2, 11))
    holidays.add(date(year, 2, 23))
    holidays.add(date(year, 3, 20))
    holidays.add(date(year, 4, 29))
    holidays.add(date(year, 5, 3))
    holidays.add(date(year, 5, 4))
    holidays.add(date(year, 5, 5))
    holidays.add(nth_weekday_of_month(year, 7, 0, 3))
    holidays.add(date(year, 8, 11))
    holidays.add(nth_weekday_of_month(year, 9, 0, 3))
    holidays.add(date(year, 9, 23))
    holidays.add(nth_weekday_of_month(year, 10, 0, 2))
    holidays.add(date(year, 11, 3))
    holidays.add(date(year, 11, 23))
    return holidays


def is_pension_day(d: date) -> bool:
    return d.month % 2 == 0 and d.day == 15


def sample_weather(month: int, rng: np.random.Generator) -> str:
    if month in (12, 1, 2, 3):
        probs = [0.30, 0.30, 0.15, 0.25]
    elif month in (6, 7):
        probs = [0.35, 0.30, 0.35, 0.0]
    else:
        probs = [0.55, 0.25, 0.18, 0.02]
    return rng.choice(["sunny", "cloudy", "rainy", "snowy"], p=probs)


def sample_temperature(d: date, rng: np.random.Generator) -> float:
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
    is_wed: bool,
    is_weekend: bool,
    is_pension: bool,
    is_holiday: bool,
    weather: str,
    temperature: float,
    cpi_index: float,
    rng: np.random.Generator,
) -> int:
    """Apply the multipliers. Caller has already ensured the product is
    available on this day."""
    multiplier = 1.0
    if is_wed:
        multiplier *= 2.0       # 水曜セール（販促効果）
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

    # Inflation drag: 1% rise in CPI reduces sales by ~0.5% (mild elasticity).
    inflation_drag = (cpi_index - 100.0) * 0.005
    multiplier *= max(0.7, 1.0 - inflation_drag)

    expected = base * multiplier
    noise = rng.normal(0, expected * 0.15)
    return int(max(0, round(expected + noise)))


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    holiday_set: set[date] = set()
    for year in range(START_DATE.year, END_DATE.year + 1):
        holiday_set |= japanese_holidays(year)

    rows: list[dict] = []
    current = START_DATE
    while current <= END_DATE:
        dow_idx = current.weekday()
        is_wed = dow_idx == 2
        is_weekend = dow_idx >= 5
        is_holiday = current in holiday_set
        is_pension = is_pension_day(current)
        is_sale = is_wed or is_weekend
        cpi_index = MONTHLY_CPI[(current.year, current.month)]

        weather = sample_weather(current.month, rng)
        temperature = sample_temperature(current, rng)
        precipitation = sample_precipitation(weather, rng)

        for product in PRODUCTS:
            available = dow_idx in product["available_dows"]
            if available:
                sales = compute_sales(
                    base=product["base"],
                    is_wed=is_wed,
                    is_weekend=is_weekend,
                    is_pension=is_pension,
                    is_holiday=is_holiday,
                    weather=weather,
                    temperature=temperature,
                    cpi_index=cpi_index,
                    rng=rng,
                )
            else:
                sales = 0
            effective_price = product["price_sale"] if is_sale else product["price_normal"]

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
                "effective_price": effective_price,
                "cpi_index": cpi_index,
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
