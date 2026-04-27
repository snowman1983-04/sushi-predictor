"""Train one model per product and persist artifacts under models/."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.feature_engineering import PRODUCT_IDS, build_features
from src.models import build

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
N_SPLITS = 5


@dataclass
class ProductMetrics:
    product_id: str
    mae: float
    rmse: float
    r2: float
    residual_std: float
    n_train: int


def _cv_metrics(pipeline_factory, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float, float]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        pipe = pipeline_factory()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        r2s.append(r2_score(y[test_idx], pred))
    return float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(r2s))


def _train_one_product(
    df_product: pd.DataFrame, model_name: str
) -> tuple[Pipeline, ProductMetrics]:
    X = build_features(df_product)
    y = df_product["sales_count"].to_numpy()

    factory = lambda: build(model_name)
    mae, rmse, r2 = _cv_metrics(factory, X, y)

    final = factory()
    final.fit(X, y)
    residuals = y - final.predict(X)
    residual_std = float(np.std(residuals, ddof=1))

    metrics = ProductMetrics(
        product_id=str(df_product["product_id"].iloc[0]),
        mae=mae,
        rmse=rmse,
        r2=r2,
        residual_std=residual_std,
        n_train=len(df_product),
    )
    return final, metrics


def train_all(model_name: str = "linear", out_dir: Path = MODELS_DIR) -> pd.DataFrame:
    df = load_data()
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for pid in PRODUCT_IDS:
        sub = df[df["product_id"] == pid].reset_index(drop=True)
        pipeline, metrics = _train_one_product(sub, model_name)

        artifact = {
            "pipeline": pipeline,
            "residual_std": metrics.residual_std,
            "product_id": pid,
            "model_name": model_name,
        }
        joblib.dump(artifact, out_dir / f"{model_name}_{pid}.pkl")
        rows.append(asdict(metrics))

    return pd.DataFrame(rows).set_index("product_id")


def main() -> None:
    metrics = train_all()
    print("=== Cross-validated metrics (5-fold TimeSeriesSplit) ===")
    print(metrics.round(3).to_string())


if __name__ == "__main__":
    main()
