"""Load trained per-product models and produce predictions with intervals."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import PRODUCT_IDS, prepare_input

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
Z_95 = 1.96

PRODUCT_NAMES = {
    "P001": "水木寿司（10貫）",
    "P002": "8貫寿司",
    "P003": "10貫寿司",
    "P004": "ランチ寿司",
}

SALE_ONLY_PRODUCTS = {"P004"}


def model_path(product_id: str, model_name: str = "linear") -> Path:
    return MODELS_DIR / f"{model_name}_{product_id}.pkl"


def load_models(model_name: str = "linear") -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    for pid in PRODUCT_IDS:
        path = model_path(pid, model_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}. Run `python -m src.trainer` first."
            )
        artifacts[pid] = joblib.load(path)
    return artifacts


def predict(
    artifacts: dict[str, dict],
    target_date: date | datetime | str,
    weather: str,
    temperature: float,
    precipitation: float,
) -> pd.DataFrame:
    """Return per-product predictions with 95% confidence intervals."""
    X = prepare_input(target_date, weather, temperature, precipitation)
    is_sale_day = bool(X["is_sale_day"].iloc[0])

    rows: list[dict] = []
    for pid in PRODUCT_IDS:
        artifact = artifacts[pid]
        pipeline = artifact["pipeline"]
        residual_std = float(artifact["residual_std"])

        if pid in SALE_ONLY_PRODUCTS and not is_sale_day:
            point = 0.0
            ci = 0.0
        else:
            raw = float(pipeline.predict(X)[0])
            point = max(0.0, raw)
            ci = Z_95 * residual_std

        lower = max(0.0, point - ci)
        upper = point + ci
        rows.append(
            {
                "product_id": pid,
                "product_name": PRODUCT_NAMES[pid],
                "predicted": round(point),
                "ci_half_width": round(ci, 1),
                "lower_95": round(lower),
                "upper_95": round(upper),
            }
        )

    return pd.DataFrame(rows)
