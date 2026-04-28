"""Train one model per (model_name, product_id) and persist artifacts.

Updated after the v1.1 review:
- Classification thresholds are fit per-fold on the training portion only
  (previously fit on the entire dataset, which leaked test-fold information).
- residual_std for regression confidence intervals is now computed from
  CV test-fold residuals (previously training-fit residuals, which were
  optimistically biased).
- Training metrics (full-data fit) are now reported alongside validation
  (CV) metrics so the gap is visible.
"""

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


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _cv_regression(factory, X: pd.DataFrame, y: np.ndarray) -> dict[str, object]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics: list[dict[str, float]] = []
    all_test_residuals: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        pipe = factory()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        fold_metrics.append(_regression_metrics(y[test_idx], pred))
        all_test_residuals.extend((y[test_idx] - pred).tolist())

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    avg["cv_residual_std"] = float(np.std(all_test_residuals, ddof=1))
    return avg


def _cv_classification_no_leak(
    factory, X: pd.DataFrame, sales: pd.Series
) -> dict[str, object]:
    """Classification CV with per-fold threshold fitting (no leak).

    The 3-class thresholds are derived from the training portion of each fold
    only. Test-fold rows are then discretized with the same thresholds. This
    prevents the test-fold sales distribution from biasing the class
    boundaries.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics: list[dict[str, float]] = []
    cm_total = np.zeros((3, 3), dtype=int)
    for train_idx, test_idx in tscv.split(X):
        sales_train = sales.iloc[train_idx]
        sales_test = sales.iloc[test_idx]
        low, high = fit_sales_thresholds(sales_train)
        y_train = discretize_sales(sales_train, low, high).to_numpy()
        y_test = discretize_sales(sales_test, low, high).to_numpy()

        pipe = factory()
        pipe.fit(X.iloc[train_idx], y_train)
        pred = pipe.predict(X.iloc[test_idx])
        fold_metrics.append(_classification_metrics(y_test, pred))
        cm_total += confusion_matrix(y_test, pred, labels=CLASS_LABELS_ORDERED)

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    avg["confusion_matrix"] = cm_total.tolist()
    return avg


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
        sales = df_product["sales_count"]
        cv_metrics = _cv_classification_no_leak(factory, X, sales)
        # Final model: fit thresholds on the full data and refit on full data,
        # because at inference time we have no fold structure. The leak
        # concern only affected evaluation, not deployment behavior.
        low, high = fit_sales_thresholds(sales)
        y_full = discretize_sales(sales, low, high).to_numpy()
        final = factory()
        final.fit(X, y_full)

        train_pred = final.predict(X)
        train_metrics = _classification_metrics(y_full, train_pred)

        metrics: dict[str, object] = dict(cv_metrics)
        metrics["training"] = train_metrics
        artifact.update({
            "pipeline": final,
            "thresholds": (low, high),
            "metrics": metrics,
            "task": "classification",
        })
    else:
        y = df_product["sales_count"].to_numpy()
        cv_metrics = _cv_regression(factory, X, y)
        final = factory()
        final.fit(X, y)

        train_pred = final.predict(X)
        train_metrics = _regression_metrics(y, train_pred)

        metrics = {k: v for k, v in cv_metrics.items() if k != "cv_residual_std"}
        metrics["training"] = train_metrics
        residual_std = float(cv_metrics["cv_residual_std"])
        artifact.update({
            "pipeline": final,
            "residual_std": residual_std,  # now CV-based
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
        m = artifact["metrics"]
        row: dict[str, object] = {"product_id": pid}
        for k, v in m.items():
            if k in ("confusion_matrix", "training"):
                continue
            row[f"cv_{k}"] = v
        for k, v in m["training"].items():
            row[f"train_{k}"] = v
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
        print(f"\n=== {name} (CV vs training) ===")
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
