"""Load trained per-product models and produce predictions."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import CLASS_LABELS, PRODUCT_IDS, prepare_input
from src.models import is_classification

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
Z_95 = 1.96

PRODUCT_NAMES = {
    "P001": "水木寿司（10貫）",
    "P002": "春8貫寿司",
    "P003": "絆10貫寿司",
    "P004": "ランチ寿司",
}

# Operational availability schedule (Mon=0..Sun=6).
PRODUCT_AVAILABLE_DOWS: dict[str, set[int]] = {
    "P001": {2, 3},                # Wed, Thu
    "P002": {0, 1, 2, 3, 4, 5, 6}, # everyday
    "P003": {0, 1, 2, 3, 4, 5, 6}, # everyday
    "P004": {0, 1, 4, 5, 6},       # Mon, Tue, Fri, Sat, Sun
}

# Price master (yen). price_sale applies on is_sale_day (Wed/weekend).
PRODUCT_PRICES: dict[str, dict[str, int]] = {
    "P001": {"normal": 500, "sale": 500},
    "P002": {"normal": 600, "sale": 500},
    "P003": {"normal": 600, "sale": 500},
    "P004": {"normal": 500, "sale": 500},
}


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


def is_product_available(product_id: str, target_date: date) -> bool:
    return target_date.weekday() in PRODUCT_AVAILABLE_DOWS[product_id]


def effective_price_for(product_id: str, is_sale_day: bool) -> int:
    prices = PRODUCT_PRICES[product_id]
    return prices["sale"] if is_sale_day else prices["normal"]


def _coerce_date(target_date: date | datetime | str) -> date:
    if isinstance(target_date, str):
        return datetime.fromisoformat(target_date).date()
    if isinstance(target_date, datetime):
        return target_date.date()
    return target_date


def _predict_regression(
    artifacts: dict[str, dict],
    target_date: date,
    weather: str,
    temperature: float,
    precipitation: float,
    cpi_index: float,
) -> pd.DataFrame:
    is_sale_day = target_date.weekday() == 2 or target_date.weekday() >= 5
    rows: list[dict] = []
    for pid in PRODUCT_IDS:
        artifact = artifacts[pid]
        if not is_product_available(pid, target_date):
            point, ci = 0.0, 0.0
        else:
            X = prepare_input(
                target_date, weather, temperature, precipitation,
                effective_price=effective_price_for(pid, is_sale_day),
                cpi_index=cpi_index,
            )
            raw = float(artifact["pipeline"].predict(X)[0])
            point = max(0.0, raw)
            ci = Z_95 * float(artifact["residual_std"])
        rows.append({
            "product_id": pid,
            "product_name": PRODUCT_NAMES[pid],
            "predicted": round(point),
            "ci_half_width": round(ci, 1),
            "lower_95": round(max(0.0, point - ci)),
            "upper_95": round(point + ci),
        })
    return pd.DataFrame(rows)


def _predict_classification(
    artifacts: dict[str, dict],
    target_date: date,
    weather: str,
    temperature: float,
    precipitation: float,
    cpi_index: float,
) -> pd.DataFrame:
    is_sale_day = target_date.weekday() == 2 or target_date.weekday() >= 5
    rows: list[dict] = []
    for pid in PRODUCT_IDS:
        artifact = artifacts[pid]
        pipeline = artifact["pipeline"]

        if not is_product_available(pid, target_date):
            label, prob = 0, 1.0
        else:
            X = prepare_input(
                target_date, weather, temperature, precipitation,
                effective_price=effective_price_for(pid, is_sale_day),
                cpi_index=cpi_index,
            )
            label = int(pipeline.predict(X)[0])
            probs = pipeline.predict_proba(X)[0]
            classes = list(pipeline.classes_)
            prob = float(probs[classes.index(label)])

        rows.append({
            "product_id": pid,
            "product_name": PRODUCT_NAMES[pid],
            "class_label": label,
            "class_name": CLASS_LABELS[label],
            "probability": round(prob, 3),
        })
    return pd.DataFrame(rows)


def predict(
    artifacts: dict[str, dict],
    target_date: date | datetime | str,
    weather: str,
    temperature: float,
    precipitation: float,
    cpi_index: float = 100.0,
) -> pd.DataFrame:
    target_date = _coerce_date(target_date)
    sample = next(iter(artifacts.values()))
    model_name = sample["model_name"]
    if is_classification(model_name):
        return _predict_classification(
            artifacts, target_date, weather, temperature, precipitation, cpi_index
        )
    return _predict_regression(
        artifacts, target_date, weather, temperature, precipitation, cpi_index
    )
