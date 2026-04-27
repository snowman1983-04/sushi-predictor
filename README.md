# 🍣 sushi-predictor

寿司商品の販売数を、日付・曜日・天気などから予測する Streamlit アプリ。
G検定取得に向けた学習およびポートフォリオを目的とした個人プロジェクトです。

> **Note**：本プロジェクトで使用するデータはすべて**ダミーデータ**です。実在する店舗の売上データは一切使用していません。

---

## 1. プロジェクト概要

過去 2 年分のダミー販売データと気象条件を入力として、複数の機械学習モデル（線形回帰・ロジスティック回帰・ランダムフォレスト・勾配ブースティング）で寿司商品別の販売数を予測する Web アプリを構築します。

データサイエンスの古典的手法を体系的に実装することで、G検定の知識定着とポートフォリオ化を目指します。

## 2. デモ

> _M2 以降で追加予定_：スクリーンショット / デモ GIF をここに掲載します。

## 3. 技術スタック

| 項目 | 採用技術 |
|---|---|
| 言語 | Python 3.11+ |
| Web フレームワーク | Streamlit |
| 機械学習 | scikit-learn, LightGBM |
| データ操作 | pandas, numpy |
| 可視化 | matplotlib, seaborn, plotly |
| バージョン管理 | Git / GitHub |

## 4. なぜこのアプローチか

- **古典統計手法を選んだ理由**：データ規模（約 2,920 行）と予測タスクの性質を考えると、線形回帰や決定木系で十分な精度が出る見込みです。解釈性が高く、特徴量の効き方を説明できる点も学習目的に合致します。
- **LLM を採用しない理由**：販売数予測は構造化された数値データに対する回帰問題であり、自然言語処理モデルの出番はありません。LLM を使うとコスト・レイテンシ・解釈性のすべてが悪化し、過剰技術になります。
- **詳細**：`docs/methodology.md` で各モデルの選定根拠と数学的背景を解説予定。

## 5. データ仕様

ダミーデータの仕様は [`spec.md`](./spec.md) のセクション 4 を参照してください。

- **期間**：2024-04-01 〜 2026-03-31（730 日）
- **商品数**：4 種類（P001〜P004）
- **総レコード数**：2,920 行
- **生成元**：[`data/generate_data.py`](./data/generate_data.py)（乱数シード固定＝42、再現可能）

カラム：`date, product_id, product_name, day_of_week, is_weekend, is_holiday, is_pension_day, is_sale_day, weather, temperature, precipitation, sales_count`

## 6. インストール・実行方法

### 必要環境
- Python 3.11 以上
- Windows / macOS / Linux

### セットアップ

```bash
# 1. リポジトリを clone
git clone https://github.com/<your-account>/sushi-predictor.git
cd sushi-predictor

# 2. 仮想環境を作成し依存パッケージをインストール
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt

# 3. ダミーデータを生成
python data/generate_data.py

# 4. Streamlit アプリを起動（M2 以降で実装）
streamlit run app.py
```

## 7. モデル比較結果

> _M3 完了時に追加予定_：MAE / RMSE / R² の比較表をここに掲載します。

## 8. ディレクトリ構成

```
sushi-predictor/
├── README.md
├── spec.md                    # プロジェクト仕様書
├── requirements.txt
├── .gitignore
├── app.py                     # Streamlit エントリーポイント（M2 で作成）
├── data/
│   ├── generate_data.py       # ダミーデータ生成スクリプト
│   └── sushi_sales.csv        # 生成データ（git 管理対象）
├── src/
│   ├── data_loader.py         # データ読み込み・前処理
│   ├── feature_engineering.py
│   ├── models.py
│   ├── trainer.py
│   └── predictor.py
├── models/                    # 学習済みモデル保存先（git 管理外）
├── notebooks/
│   └── exploration.ipynb      # EDA
└── docs/
    ├── methodology.md         # 手法解説（G検対策）
    ├── model_comparison.md    # モデル比較レポート
    └── images/
```

## 9. 学んだこと

> _M5 完了時に追加予定_：G検対策として整理した観点（バイアス-バリアンス、交差検証、特徴量エンジニアリング 等）を記載します。

## 10. ライセンス

MIT License

---

**ステータス**：🚧 開発中（M1 データ基盤完了）
**仕様書**：[`spec.md`](./spec.md)
