"""Streamlit entry point for the sushi sales predictor (M3 scope).

Tabs 1 (prediction) and 3 (model comparison) are wired. Tabs 2/4 are stubs.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_data
from src.feature_engineering import (
    CLASS_LABELS,
    PRODUCT_IDS,
    build_features,
    discretize_sales,
)
from src.models import MODEL_DISPLAY_NAMES, MODEL_REGISTRY, is_classification
from src.predictor import PRODUCT_NAMES, load_models, model_path, predict
from src.trainer import train_all

st.set_page_config(page_title="寿司販売数予測アプリ", page_icon="🍣", layout="wide")


@st.cache_data
def _cached_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def _cached_models(model_name: str) -> dict[str, dict]:
    if not model_path("P001", model_name).exists():
        with st.spinner(f"初回起動：{MODEL_DISPLAY_NAMES[model_name]} を学習しています..."):
            train_all(model_name=model_name)
    return load_models(model_name)


def _data_summary(df: pd.DataFrame) -> None:
    st.metric("総レコード数", f"{len(df):,}")
    st.metric("期間", f"{df['date'].min().date()} 〜 {df['date'].max().date()}")
    st.metric("商品数", df["product_id"].nunique())
    st.caption("商品別の平均販売数：")
    by_product = (
        df.groupby("product_id")["sales_count"].mean().round(1).rename("平均/日").to_frame()
    )
    st.dataframe(by_product, use_container_width=True)


def _prediction_inputs():
    col1, col2 = st.columns(2)
    with col1:
        target_date = st.date_input("日付", value=date(2026, 5, 1))
        weather = st.selectbox(
            "天気",
            options=["sunny", "cloudy", "rainy", "snowy"],
            format_func=lambda x: {
                "sunny": "☀️ 晴れ",
                "cloudy": "☁️ 曇り",
                "rainy": "🌧 雨",
                "snowy": "❄️ 雪",
            }[x],
        )
    with col2:
        temperature = st.slider("気温（℃）", min_value=-15, max_value=35, value=20)
        precipitation = st.slider("降水量（mm）", min_value=0, max_value=30, value=0)
    return target_date, weather, temperature, precipitation


def _prediction_tab(model_name: str, models: dict[str, dict]) -> None:
    st.subheader(f"販売数を予測（{MODEL_DISPLAY_NAMES[model_name]}）")
    target_date, weather, temperature, precipitation = _prediction_inputs()

    if st.button("予測実行", type="primary"):
        result = predict(models, target_date, weather, float(temperature), float(precipitation))
        st.success(f"{target_date} の予測結果")

        if is_classification(model_name):
            display = result[["product_name", "class_name", "probability"]].rename(columns={
                "product_name": "商品",
                "class_name": "予測クラス",
                "probability": "確率",
            })
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption("クラス：少 / 普 / 多 は商品ごとの 33/66 パーセンタイルで分割。")
        else:
            display = result[["product_name", "predicted", "ci_half_width", "lower_95", "upper_95"]].rename(columns={
                "product_name": "商品",
                "predicted": "予測販売数",
                "ci_half_width": "±",
                "lower_95": "95%下限",
                "upper_95": "95%上限",
            })
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption("信頼区間 = 学習データ残差σ × 1.96 の簡易版。")


# ---------- Tab3: model comparison ---------------------------------------------------

REGRESSION_MODELS = [m for m in MODEL_REGISTRY if not is_classification(m)]
CLASSIFICATION_MODELS = [m for m in MODEL_REGISTRY if is_classification(m)]


def _regression_metrics_table() -> pd.DataFrame:
    rows: list[dict] = []
    for name in REGRESSION_MODELS:
        artifacts = _cached_models(name)
        for pid in PRODUCT_IDS:
            m = artifacts[pid]["metrics"]
            rows.append({
                "モデル": MODEL_DISPLAY_NAMES[name],
                "商品": PRODUCT_NAMES[pid],
                "MAE": round(m["mae"], 3),
                "RMSE": round(m["rmse"], 3),
                "R²": round(m["r2"], 3),
            })
    return pd.DataFrame(rows)


def _classification_metrics_table() -> pd.DataFrame:
    rows: list[dict] = []
    for name in CLASSIFICATION_MODELS:
        artifacts = _cached_models(name)
        for pid in PRODUCT_IDS:
            m = artifacts[pid]["metrics"]
            rows.append({
                "モデル": MODEL_DISPLAY_NAMES[name],
                "商品": PRODUCT_NAMES[pid],
                "正解率": round(m["accuracy"], 3),
                "適合率(macro)": round(m["precision_macro"], 3),
                "再現率(macro)": round(m["recall_macro"], 3),
                "F1(macro)": round(m["f1_macro"], 3),
            })
    return pd.DataFrame(rows)


def _feature_importance(model_name: str, artifacts: dict[str, dict]) -> pd.DataFrame | None:
    rows: list[dict] = []
    for pid in PRODUCT_IDS:
        pipe = artifacts[pid]["pipeline"]
        sklearn_model = pipe.steps[-1][1]
        try:
            feature_names = pipe.feature_names_in_
        except AttributeError:
            feature_names = None

        if hasattr(sklearn_model, "feature_importances_"):
            importances = sklearn_model.feature_importances_
        elif hasattr(sklearn_model, "coef_"):
            coef = sklearn_model.coef_
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return None

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]
        for fname, imp in zip(feature_names, importances):
            rows.append({"商品": PRODUCT_NAMES[pid], "特徴量": fname, "重要度": float(imp)})
    return pd.DataFrame(rows)


def _confusion_matrix_for(name: str, pid: str) -> np.ndarray:
    cm = _cached_models(name)[pid]["metrics"]["confusion_matrix"]
    return np.array(cm)


def _comparison_tab() -> None:
    st.subheader("回帰モデル比較")
    reg_table = _regression_metrics_table()
    st.dataframe(reg_table, use_container_width=True, hide_index=True)
    st.caption("MAE/RMSEは小さいほど良い、R²は1に近いほど良い。5-fold TimeSeriesSplitで評価。")

    pivot_mae = reg_table.pivot(index="商品", columns="モデル", values="MAE")
    fig = px.bar(pivot_mae, barmode="group", title="商品×モデル別 MAE", labels={"value": "MAE", "商品": "商品"})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("特徴量重要度")
    selected_reg_model = st.selectbox(
        "モデルを選択",
        options=REGRESSION_MODELS,
        format_func=lambda x: MODEL_DISPLAY_NAMES[x],
        key="fi_model",
    )
    fi_df = _feature_importance(selected_reg_model, _cached_models(selected_reg_model))
    if fi_df is not None:
        avg = fi_df.groupby("特徴量")["重要度"].mean().sort_values(ascending=True)
        fig_fi = px.bar(avg, orientation="h", title=f"{MODEL_DISPLAY_NAMES[selected_reg_model]} の特徴量重要度（4商品平均）",
                        labels={"value": "重要度", "特徴量": "特徴量"})
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("線形回帰は係数の絶対値、RFとGBはモデルの feature_importances_ を表示。"
                   "スケールはモデルごとに異なるので、相対比較に注意。")

    st.divider()
    st.subheader("分類モデル（ロジスティック回帰）")
    cls_table = _classification_metrics_table()
    st.dataframe(cls_table, use_container_width=True, hide_index=True)
    st.caption("販売数を 33/66 パーセンタイルで 少/普/多 に離散化したクラスを予測。")

    selected_pid = st.selectbox(
        "混同行列を見たい商品",
        options=PRODUCT_IDS,
        format_func=lambda x: PRODUCT_NAMES[x],
        key="cm_product",
    )
    cm = _confusion_matrix_for("logreg", selected_pid)
    cm_df = pd.DataFrame(
        cm,
        index=[f"実際: {CLASS_LABELS[i]}" for i in range(3)],
        columns=[f"予測: {CLASS_LABELS[i]}" for i in range(3)],
    )
    st.dataframe(cm_df, use_container_width=True)
    st.caption("対角成分が多いほど良い。行=実際のクラス、列=予測したクラス。")


# ---------- Main ---------------------------------------------------------------------


def main() -> None:
    st.title("🍣 寿司販売数予測アプリ")
    st.caption("ダミーデータによる学習用ポートフォリオ（G検定対策）")

    with st.sidebar:
        st.header("設定")
        model_name = st.radio(
            "モデル選択",
            options=list(MODEL_REGISTRY.keys()),
            format_func=lambda x: MODEL_DISPLAY_NAMES[x],
        )
        st.divider()
        st.header("データ統計")
        _data_summary(_cached_data())

    models = _cached_models(model_name)

    tab1, tab2, tab3, tab4 = st.tabs(["予測", "データ探索", "モデル比較", "手法解説"])
    with tab1:
        _prediction_tab(model_name, models)
    with tab2:
        st.info("M4で実装予定：時系列・曜日別・天気別グラフ、相関ヒートマップ。")
    with tab3:
        _comparison_tab()
    with tab4:
        st.info("M4で実装予定：各モデルの数学的説明、選択根拠、LLM不採用の理由。")


if __name__ == "__main__":
    main()
