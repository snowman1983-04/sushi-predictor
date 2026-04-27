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


def _predict_regression(
    artifacts: dict[str, dict],
    X: pd.DataFrame,
    is_sale_day: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for pid in PRODUCT_IDS:
        artifact = artifacts[pid]
        if pid in SALE_ONLY_PRODUCTS and not is_sale_day:
            point, ci = 0.0, 0.0
        else:
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
    X: pd.DataFrame,
    is_sale_day: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for pid in PRODUCT_IDS:
        artifact = artifacts[pid]
        pipeline = artifact["pipeline"]

        if pid in SALE_ONLY_PRODUCTS and not is_sale_day:
            label, prob = 0, 1.0
        else:
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
) -> pd.DataFrame:
    X = prepare_input(target_date, weather, temperature, precipitation)
    is_sale_day = bool(X["is_sale_day"].iloc[0])

    sample = next(iter(artifacts.values()))
    model_name = sample["model_name"]
    if is_classification(model_name):
        return _predict_classification(artifacts, X, is_sale_day)
    return _predict_regression(artifacts, X, is_sale_day)
