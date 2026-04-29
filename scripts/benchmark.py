"""Benchmark MLモデルとナイーブベースラインを商品別に比較する。

既存の trainer._cv_regression / _cv_classification_no_leak をそのまま使い、
さらに以下のベースラインを TimeSeriesSplit で同条件評価する：

  Regression baselines:
    - mean        : 学習foldの sales_count 平均で全予測
    - dow_mean    : 学習foldの曜日別平均で予測
    - last_week   : 7日前の値（学習fold末尾もしくはtest fold内のラグ）

  Classification baselines:
    - majority    : 学習fold多数派クラスを全予測
    - dow_mode    : 学習fold内 曜日別最頻クラスで予測（無ければ全体最頻）

CSV と JSON で benchmarks/ 配下に保存。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_dataset
from src.feature_engineering import (
    build_features,
    discretize_sales,
    fit_sales_thresholds,
)
from src.models import build
from src.trainer import (
    N_SPLITS,
    _classification_metrics,
    _cv_classification_no_leak,
    _cv_regression,
    _regression_metrics,
)

OUT_DIR = ROOT / "benchmarks"


def _baseline_regression(df_product: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Run regression baselines under the same TimeSeriesSplit folds as ML models."""
    df = df_product.sort_values("date").reset_index(drop=True)
    y = df["sales_count"].to_numpy()
    dow = df["date"].dt.dayofweek.to_numpy()

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics: dict[str, list[dict[str, float]]] = {
        "mean": [],
        "dow_mean": [],
        "last_week": [],
    }
    for train_idx, test_idx in tscv.split(df):
        y_train, y_test = y[train_idx], y[test_idx]
        dow_train, dow_test = dow[train_idx], dow[test_idx]

        # mean baseline
        pred_mean = np.full_like(y_test, fill_value=float(y_train.mean()), dtype=float)
        fold_metrics["mean"].append(_regression_metrics(y_test, pred_mean))

        # day-of-week mean baseline
        dow_avg = pd.Series(y_train).groupby(dow_train).mean()
        global_mean = float(y_train.mean())
        pred_dow = np.array(
            [float(dow_avg.get(d, global_mean)) for d in dow_test], dtype=float
        )
        fold_metrics["dow_mean"].append(_regression_metrics(y_test, pred_dow))

        # last_week (lag-7) baseline: use y[i-7] from the full series
        # If i-7 lies before the training fold start, fall back to y_train mean.
        preds_lw = []
        for i in test_idx:
            if i - 7 >= 0:
                preds_lw.append(float(y[i - 7]))
            else:
                preds_lw.append(global_mean)
        pred_lw = np.array(preds_lw, dtype=float)
        fold_metrics["last_week"].append(_regression_metrics(y_test, pred_lw))

    out: dict[str, dict[str, float]] = {}
    for name, folds in fold_metrics.items():
        out[name] = {k: float(np.mean([m[k] for m in folds])) for k in folds[0]}
    return out


def _baseline_classification(df_product: pd.DataFrame) -> dict[str, dict[str, float]]:
    df = df_product.sort_values("date").reset_index(drop=True)
    sales = df["sales_count"]
    dow = df["date"].dt.dayofweek.to_numpy()

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics: dict[str, list[dict[str, float]]] = {
        "majority": [],
        "dow_mode": [],
    }
    for train_idx, test_idx in tscv.split(df):
        s_train = sales.iloc[train_idx]
        s_test = sales.iloc[test_idx]
        low, high = fit_sales_thresholds(s_train)
        y_train = discretize_sales(s_train, low, high).to_numpy()
        y_test = discretize_sales(s_test, low, high).to_numpy()
        dow_train, dow_test = dow[train_idx], dow[test_idx]

        # majority class
        majority = int(pd.Series(y_train).mode().iloc[0])
        pred_maj = np.full_like(y_test, fill_value=majority)
        fold_metrics["majority"].append(_classification_metrics(y_test, pred_maj))

        # dow mode
        dow_mode = (
            pd.Series(y_train).groupby(dow_train).agg(lambda s: int(s.mode().iloc[0]))
        )
        pred_dow = np.array(
            [int(dow_mode.get(d, majority)) for d in dow_test], dtype=int
        )
        fold_metrics["dow_mode"].append(_classification_metrics(y_test, pred_dow))

    out: dict[str, dict[str, float]] = {}
    for name, folds in fold_metrics.items():
        out[name] = {k: float(np.mean([m[k] for m in folds])) for k in folds[0]}
    return out


def _ml_regression(df_product: pd.DataFrame, model_name: str) -> dict[str, float]:
    X = build_features(df_product)
    y = df_product["sales_count"].to_numpy()
    factory = lambda: build(model_name)
    cv = _cv_regression(factory, X, y)
    return {k: v for k, v in cv.items() if k != "cv_residual_std"}


def _ml_classification(df_product: pd.DataFrame, model_name: str) -> dict[str, float]:
    X = build_features(df_product)
    sales = df_product["sales_count"]
    factory = lambda: build(model_name)
    cv = _cv_classification_no_leak(factory, X, sales)
    return {k: v for k, v in cv.items() if k != "confusion_matrix"}


def _unify_products(df: pd.DataFrame, unified_id: str) -> pd.DataFrame:
    """Collapse multi-SKU rows into a single date-indexed series.

    sales_count is summed; effective_price is sales-weighted (falls back to a
    plain mean when total count is zero); calendar/weather columns are taken
    from the first row of each date since they are identical across SKUs in
    the source data.
    """
    df = df.sort_values(["date", "product_id"]).reset_index(drop=True)
    revenue = df["effective_price"] * df["sales_count"]
    grouped = df.groupby("date", as_index=False)
    sales = grouped["sales_count"].sum()
    rev = revenue.groupby(df["date"]).sum().reindex(sales["date"]).to_numpy()
    count = sales["sales_count"].to_numpy()
    weighted_price = np.where(count > 0, rev / np.where(count > 0, count, 1), 0.0)
    fallback_price = grouped["effective_price"].mean()["effective_price"].to_numpy()
    eff_price = np.where(count > 0, weighted_price, fallback_price)

    first = grouped.first()
    out = pd.DataFrame({
        "date": sales["date"],
        "product_id": unified_id,
        "product_name": unified_id,
        "day_of_week": first["day_of_week"],
        "is_weekend": first["is_weekend"],
        "is_holiday": first["is_holiday"],
        "is_pension_day": first["is_pension_day"],
        "is_sale_day": first["is_sale_day"],
        "weather": first["weather"],
        "temperature": first["temperature"],
        "precipitation": first["precipitation"],
        "effective_price": eff_price,
        "cpi_index": first["cpi_index"],
        "sales_count": sales["sales_count"],
    })
    return out


def run(dataset: str, unify_products: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load_dataset(dataset)
    if unify_products:
        unified_id = dataset.upper()
        df = _unify_products(df, unified_id)
    product_ids = df["product_id"].drop_duplicates().tolist()

    reg_rows: list[dict] = []
    cls_rows: list[dict] = []
    raw: dict = {"dataset": dataset, "products": {}}

    for pid in product_ids:
        sub = df[df["product_id"] == pid].reset_index(drop=True)
        raw["products"][pid] = {"n_rows": len(sub)}

        # Regression
        reg_baselines = _baseline_regression(sub)
        for bname, m in reg_baselines.items():
            reg_rows.append({"product_id": pid, "model": f"baseline_{bname}", **m})
        for mname in ("linear", "rf", "gb"):
            m = _ml_regression(sub, mname)
            reg_rows.append({"product_id": pid, "model": mname, **m})

        # Classification
        cls_baselines = _baseline_classification(sub)
        for bname, m in cls_baselines.items():
            cls_rows.append({"product_id": pid, "model": f"baseline_{bname}", **m})
        m = _ml_classification(sub, "logreg")
        cls_rows.append({"product_id": pid, "model": "logreg", **m})

        raw["products"][pid]["regression"] = {
            r["model"]: {k: r[k] for k in ("mae", "rmse", "r2")}
            for r in reg_rows
            if r["product_id"] == pid
        }
        raw["products"][pid]["classification"] = {
            r["model"]: {
                k: r[k]
                for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
            }
            for r in cls_rows
            if r["product_id"] == pid
        }

    reg_df = pd.DataFrame(reg_rows).set_index(["product_id", "model"])
    cls_df = pd.DataFrame(cls_rows).set_index(["product_id", "model"])
    return reg_df, cls_df, raw


def _summary_table(reg_df: pd.DataFrame, cls_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model average across products (macro view)."""
    reg_avg = reg_df.groupby(level="model")[["mae", "rmse", "r2"]].mean()
    cls_avg = cls_df.groupby(level="model")[
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    ].mean()
    return reg_avg, cls_avg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["sushi", "bento"],
        default="sushi",
        help="Which dataset adapter to load (default: sushi)",
    )
    parser.add_argument(
        "--unify-products",
        action="store_true",
        help="Collapse multi-SKU rows into a single date-indexed series "
        "(sales summed, prices sales-weighted) before benchmarking.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.dataset}_unified" if args.unify_products else args.dataset

    reg_df, cls_df, raw = run(args.dataset, unify_products=args.unify_products)
    reg_avg, cls_avg = _summary_table(reg_df, cls_df)

    reg_df.round(4).to_csv(OUT_DIR / f"{tag}_regression_{timestamp}.csv")
    cls_df.round(4).to_csv(OUT_DIR / f"{tag}_classification_{timestamp}.csv")
    reg_avg.round(4).to_csv(OUT_DIR / f"{tag}_regression_avg_{timestamp}.csv")
    cls_avg.round(4).to_csv(OUT_DIR / f"{tag}_classification_avg_{timestamp}.csv")

    # also overwrite "latest" pointers for easy diffing per dataset
    reg_df.round(4).to_csv(OUT_DIR / f"{tag}_regression_latest.csv")
    cls_df.round(4).to_csv(OUT_DIR / f"{tag}_classification_latest.csv")
    reg_avg.round(4).to_csv(OUT_DIR / f"{tag}_regression_avg_latest.csv")
    cls_avg.round(4).to_csv(OUT_DIR / f"{tag}_classification_avg_latest.csv")

    with (OUT_DIR / f"{tag}_raw_{timestamp}.json").open("w") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"\n### dataset = {tag} (n_rows={sum(p['n_rows'] for p in raw['products'].values())})")
    print("\n=== Regression: per-product × model (CV averages) ===")
    print(reg_df.round(3).to_string())
    print("\n=== Regression: per-model averaged across products ===")
    print(reg_avg.round(3).to_string())
    print("\n=== Classification: per-product × model (CV averages) ===")
    print(cls_df.round(3).to_string())
    print("\n=== Classification: per-model averaged across products ===")
    print(cls_avg.round(3).to_string())
    print(f"\nResults written to {OUT_DIR}/{tag}_*_{timestamp}.csv")


if __name__ == "__main__":
    main()
