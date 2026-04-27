"""Model definitions. M2 ships only linear regression; M3 will add more."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_linear_regression() -> Pipeline:
    """Standard-scale features, then fit ordinary least squares."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


MODEL_REGISTRY = {
    "linear": build_linear_regression,
}


def build(model_name: str) -> Pipeline:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}. Known: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name]()
