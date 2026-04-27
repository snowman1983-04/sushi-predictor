"""Train one model per (model_name, product_id) and persist artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.feature_engineering import (
    PRODUCT_IDS,
    build_features,
    discretize_sales,
    fit_sales_thresholds,
)
from src.models import MODEL_REGISTRY, build, is_classification

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
N_SPLITS = 5
CLASS_LABELS_ORDERED = [0, 1, 2]


def _cv_regression(factory, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in tscv.split(X):
        pipe = factory()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        r2s.append(r2_score(y[test_idx], pred))
    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s)),
    }


def _cv_classification(factory, X: pd.DataFrame, y: np.ndarray) -> dict[str, object]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs, precs, recs, f1s = [], [], [], []
    cm_total = np.zeros((3, 3), dtype=int)
    for train_idx, test_idx in tscv.split(X):
        pipe = factory()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        precs.append(precision_score(y[test_idx], pred, average="macro", zero_division=0))
        recs.append(recall_score(y[test_idx], pred, average="macro", zero_division=0))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
        cm_total += confusion_matrix(y[test_idx], pred, labels=CLASS_LABELS_ORDERED)
    return {
        "accuracy": float(np.mean(accs)),
        "precision_macro": float(np.mean(precs)),
        "recall_macro": float(np.mean(recs)),
        "f1_macro": float(np.mean(f1s)),
        "confusion_matrix": cm_total.tolist(),
    }


def _train_one(
    df_product: pd.DataFrame, model_name: str
) -> tuple[Pipeline, dict[str, object]]:
    pid = str(df_product["product_id"].iloc[0])
    X = build_features(df_product)
    factory = lambda: build(model_name)

    artifact: dict[str, object] = {
        "product_id": pid,
        "model_name": model_name,
        "n_train": len(df_product),
    }

    if is_classification(model_name):
        low, high = fit_sales_thresholds(df_product["sales_count"])
        y = discretize_sales(df_product["sales_count"], low, high).to_numpy()
        metrics = _cv_classification(factory, X, y)
        final = factory()
        final.fit(X, y)
        artifact.update({
            "pipeline": final,
            "thresholds": (low, high),
            "metrics": metrics,
            "task": "classification",
        })
    else:
        y = df_product["sales_count"].to_numpy()
        metrics = _cv_regression(factory, X, y)
        final = factory()
        final.fit(X, y)
        residual_std = float(np.std(y - final.predict(X), ddof=1))
        artifact.update({
            "pipeline": final,
            "residual_std": residual_std,
            "metrics": metrics,
            "task": "regression",
        })
    return final, artifact


def train_all(model_name: str = "linear", out_dir: Path = MODELS_DIR) -> pd.DataFrame:
    df = load_data().sort_values(["product_id", "date"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for pid in PRODUCT_IDS:
        sub = df[df["product_id"] == pid].reset_index(drop=True)
        _, artifact = _train_one(sub, model_name)
        joblib.dump(artifact, out_dir / f"{model_name}_{pid}.pkl")
        row = {"product_id": pid, **artifact["metrics"]}
        if "confusion_matrix" in row:
            row.pop("confusion_matrix")
        rows.append(row)
    return pd.DataFrame(rows).set_index("product_id")


def train_every_model(out_dir: Path = MODELS_DIR) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    for name in MODEL_REGISTRY:
        summaries[name] = train_all(model_name=name, out_dir=out_dir)
    return summaries


def main() -> None:
    summaries = train_every_model()
    for name, summary in summaries.items():
        print(f"\n=== {name} ===")
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
