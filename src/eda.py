"""Visualization helpers for Tab2 (data exploration) and Tab3 (residuals)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.feature_engineering import DOW_NAMES, WEATHER_CATEGORIES, build_features
from src.predictor import PRODUCT_NAMES, load_models

CORRELATION_COLS = [
    "sales_count",
    "temperature",
    "precipitation",
    "is_weekend",
    "is_holiday",
    "is_pension_day",
    "is_sale_day",
]


def _with_product_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["商品"] = out["product_id"].map(PRODUCT_NAMES)
    return out


def chart_timeseries(df: pd.DataFrame) -> go.Figure:
    work = _with_product_name(df)
    fig = px.line(
        work,
        x="date",
        y="sales_count",
        color="商品",
        title="商品別 販売数の推移（日次）",
        labels={"date": "日付", "sales_count": "販売数"},
    )
    fig.update_layout(hovermode="x unified")
    return fig


def chart_by_dayofweek(df: pd.DataFrame) -> go.Figure:
    work = _with_product_name(df)
    work["day_of_week"] = pd.Categorical(work["day_of_week"], categories=DOW_NAMES, ordered=True)
    work = work.sort_values("day_of_week")
    fig = px.box(
        work,
        x="day_of_week",
        y="sales_count",
        color="商品",
        title="曜日別 販売数の分布（箱ひげ図）",
        labels={"day_of_week": "曜日", "sales_count": "販売数"},
    )
    return fig


def chart_by_weather(df: pd.DataFrame) -> go.Figure:
    work = _with_product_name(df)
    agg = (
        work.groupby(["weather", "商品"], as_index=False)["sales_count"]
        .mean()
        .round(2)
    )
    agg["weather"] = pd.Categorical(agg["weather"], categories=WEATHER_CATEGORIES, ordered=True)
    agg = agg.sort_values("weather")
    fig = px.bar(
        agg,
        x="weather",
        y="sales_count",
        color="商品",
        barmode="group",
        title="天気別 平均販売数",
        labels={"weather": "天気", "sales_count": "平均販売数"},
    )
    return fig


def chart_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    sub = df[CORRELATION_COLS].copy()
    for col in sub.columns:
        if sub[col].dtype == bool:
            sub[col] = sub[col].astype(int)
    corr = sub.corr().round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="特徴量の相関ヒートマップ（数値+フラグ）",
    )
    return fig


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["sales_count", "temperature", "precipitation"]
    stats = df[numeric_cols].describe().round(2)
    stats.index = ["件数", "平均", "標準偏差", "最小", "25%", "中央値", "75%", "最大"]
    return stats


def stats_by_product(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("product_id")["sales_count"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
    )
    out.index = [PRODUCT_NAMES[pid] for pid in out.index]
    out.columns = ["日数", "平均", "標準偏差", "最小", "中央値", "最大"]
    return out


def residual_data(df_full: pd.DataFrame, model_name: str, product_id: str) -> pd.DataFrame:
    """Compute predictions on training data for residual analysis."""
    artifacts = load_models(model_name)
    sub = df_full[df_full["product_id"] == product_id].sort_values("date").reset_index(drop=True)
    X = build_features(sub)
    pipeline = artifacts[product_id]["pipeline"]
    pred = pipeline.predict(X)
    return pd.DataFrame({
        "date": sub["date"],
        "actual": sub["sales_count"].astype(float),
        "predicted": np.asarray(pred, dtype=float),
        "residual": sub["sales_count"].astype(float) - np.asarray(pred, dtype=float),
    })


def chart_residuals(resid_df: pd.DataFrame, title: str) -> go.Figure:
    """Actual-vs-predicted scatter with the y=x identity line."""
    lo = float(min(resid_df["actual"].min(), resid_df["predicted"].min()))
    hi = float(max(resid_df["actual"].max(), resid_df["predicted"].max()))
    fig = px.scatter(
        resid_df,
        x="actual",
        y="predicted",
        opacity=0.5,
        title=title,
        labels={"actual": "実測値", "predicted": "予測値"},
        hover_data={"date": True, "residual": ":.2f"},
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="理想（y=x）",
            line=dict(color="red", dash="dash"),
        )
    )
    return fig
