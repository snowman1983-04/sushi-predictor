"""Streamlit entry point for the sushi sales predictor (M2 scope).

Tab1 (prediction) is fully wired. Tabs 2-4 are placeholders to be filled in M3+.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.data_loader import load_data
from src.predictor import load_models, predict, model_path
from src.trainer import train_all

st.set_page_config(page_title="寿司販売数予測アプリ", page_icon="🍣", layout="wide")


@st.cache_data
def _cached_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def _cached_models(model_name: str) -> dict[str, dict]:
    if not model_path("P001", model_name).exists():
        with st.spinner("初回起動：モデルを学習しています..."):
            train_all(model_name=model_name)
    return load_models(model_name)


def _data_summary(df: pd.DataFrame) -> None:
    st.metric("総レコード数", f"{len(df):,}")
    st.metric("期間", f"{df['date'].min().date()} 〜 {df['date'].max().date()}")
    st.metric("商品数", df["product_id"].nunique())
    st.caption("商品別の平均販売数：")
    by_product = (
        df.groupby("product_id")["sales_count"]
        .mean()
        .round(1)
        .rename("平均/日")
        .to_frame()
    )
    st.dataframe(by_product, use_container_width=True)


def _prediction_tab(models: dict[str, dict]) -> None:
    st.subheader("販売数を予測")

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

    if st.button("予測実行", type="primary"):
        result = predict(models, target_date, weather, float(temperature), float(precipitation))

        st.success(f"{target_date} の予測結果")
        display = result[["product_name", "predicted", "ci_half_width", "lower_95", "upper_95"]]
        display = display.rename(
            columns={
                "product_name": "商品",
                "predicted": "予測販売数",
                "ci_half_width": "±",
                "lower_95": "95%下限",
                "upper_95": "95%上限",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.caption(
            "※ 信頼区間は学習データの残差標準偏差 σ から ±1.96σ で算出した簡易版。"
            "M3 以降で予測区間の厳密版に差し替え予定。"
        )


def main() -> None:
    st.title("🍣 寿司販売数予測アプリ")
    st.caption("ダミーデータによる学習用ポートフォリオ（G検定対策）")

    with st.sidebar:
        st.header("設定")
        model_name = st.radio(
            "モデル選択",
            options=["linear"],
            format_func=lambda x: {"linear": "線形回帰"}[x],
        )
        st.divider()
        st.header("データ統計")
        _data_summary(_cached_data())

    models = _cached_models(model_name)

    tab1, tab2, tab3, tab4 = st.tabs(["予測", "データ探索", "モデル評価", "手法解説"])
    with tab1:
        _prediction_tab(models)
    with tab2:
        st.info("M3で実装予定：時系列・曜日別・天気別グラフ、相関ヒートマップ。")
    with tab3:
        st.info("M3で実装予定：複数モデル比較、特徴量重要度、残差プロット。")
    with tab4:
        st.info("M4で実装予定：各モデルの数学的説明、選択根拠、LLM不採用の理由。")


if __name__ == "__main__":
    main()
