"""Model definitions for the registry consumed by trainer/predictor."""

from __future__ import annotations

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_linear_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_random_forest() -> Pipeline:
    return Pipeline(
        steps=[
            ("model", RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )


def build_gradient_boosting() -> Pipeline:
    return Pipeline(
        steps=[
            ("model", LGBMRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )),
        ]
    )


def build_logistic_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]
    )


MODEL_REGISTRY = {
    "linear": build_linear_regression,
    "rf": build_random_forest,
    "gb": build_gradient_boosting,
    "logreg": build_logistic_regression,
}

TASK_TYPE = {
    "linear": "regression",
    "rf": "regression",
    "gb": "regression",
    "logreg": "classification",
}

MODEL_DISPLAY_NAMES = {
    "linear": "線形回帰",
    "rf": "ランダムフォレスト",
    "gb": "勾配ブースティング (LightGBM)",
    "logreg": "ロジスティック回帰",
}


def build(model_name: str) -> Pipeline:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}. Known: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name]()


def is_classification(model_name: str) -> bool:
    return TASK_TYPE[model_name] == "classification"
