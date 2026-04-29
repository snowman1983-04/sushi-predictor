# sushi 1系列化ベンチ — bento と同条件で再評価

**作成日時**: 2026-04-29 12:46 JST

`scripts/benchmark.py` に `--unify-products` オプションを追加し、仮想データ sushi（4 SKU）を日付で集約した単一系列 (`product_id="SUSHI"`) として bento と同条件でベンチマークした。結果は仮想データ側が圧倒的に高スコア（rf R² 0.85 / logreg F1 0.72）。bento は rf R² が負（−0.57）。**この差は手法の優劣ではなく、データ生成プロセスの定常性に起因する**ことが確認できた。本走査で「合成ベンチは over-optimistic」という事実が定量的に裏付けられた点を記録する。

---

## 1. 背景と動機

bento ベンチは `BENTO` 単一系列で `name`（メニュー名）を意図的に特徴量化せず、日付＋天気だけのフェアな下限を測る設計（`src/datasets/bento.py:7`）。一方 sushi は 4 SKU × 1095 日 = 4380 行を商品別にループしてベンチしてきた。**「商品別ループ」と「単一系列」では CV の単位が違うため、bento と数字を直接並べる根拠が薄い**。

そこで sushi も 4 SKU を日付で sum 集約し、`product_id="SUSHI"` の 1 系列にしてからベンチを回した。

---

## 2. 実施した変更

### 2-1. `scripts/benchmark.py` に集約ロジックを追加

**新規関数** `_unify_products()`：

| 列 | 集約方法 | 理由 |
|---|---|---|
| `sales_count` | sum | 商品横断の総需要 |
| `effective_price` | 売上加重平均 (Σpx・count / Σcount) | 「平均単価」として自然。count=0 はSKU平均にフォールバック |
| `day_of_week`, `is_weekend`, `is_holiday`, `is_pension_day`, `is_sale_day` | first | 同日内で全SKU共通（データで確認済み） |
| `weather`, `temperature`, `precipitation`, `cpi_index` | first | 同上 |
| `product_id`, `product_name` | 定数 (`"SUSHI"`) | 1 系列化のため |

**CLI**：`--unify-products` フラグ追加。出力タグも `sushi_unified` に切り替え、既存の `sushi_*_latest.csv` と分離して bento と並べやすくした。

```python
# scripts/benchmark.py:150
def _unify_products(df: pd.DataFrame, unified_id: str) -> pd.DataFrame:
    df = df.sort_values(["date", "product_id"]).reset_index(drop=True)
    revenue = df["effective_price"] * df["sales_count"]
    grouped = df.groupby("date", as_index=False)
    sales = grouped["sales_count"].sum()
    rev = revenue.groupby(df["date"]).sum().reindex(sales["date"]).to_numpy()
    count = sales["sales_count"].to_numpy()
    weighted_price = np.where(count > 0, rev / np.where(count > 0, count, 1), 0.0)
    fallback_price = grouped["effective_price"].mean()["effective_price"].to_numpy()
    eff_price = np.where(count > 0, weighted_price, fallback_price)
    ...
```

---

## 3. 計測結果

### 3-1. sushi_unified（仮想データを 1 系列化）

n_rows = 1095（365 × 3 年）

| Model | MAE | RMSE | R² |
|---|---|---|---|
| baseline_mean | 15.55 | 20.54 | −0.03 |
| baseline_dow_mean | 8.78 | 12.68 | 0.60 |
| baseline_last_week | 11.71 | 16.41 | 0.34 |
| linear | 9.96 | 13.57 | 0.10 |
| **rf** | **5.06** | **7.59** | **0.85** |
| gb | 6.05 | 9.15 | 0.79 |

| Model | Accuracy | Precision_macro | Recall_macro | F1_macro |
|---|---|---|---|---|
| baseline_majority | 0.377 | 0.126 | 0.333 | 0.180 |
| baseline_dow_mode | 0.673 | 0.655 | 0.659 | 0.627 |
| **logreg** | **0.757** | **0.745** | **0.740** | **0.716** |

### 3-2. bento（実データ、既存 latest）

| Model | MAE | RMSE | R² |
|---|---|---|---|
| baseline_mean | 35.00 | 38.19 | −3.43 |
| baseline_dow_mean | 35.43 | 38.30 | −3.48 |
| baseline_last_week | 21.81 | 27.89 | −1.22 |
| linear | 23.17 | 29.02 | −1.31 |
| **rf** | **17.84** | **22.97** | **−0.57** |
| gb | 19.32 | 24.10 | −0.64 |

| Model | Accuracy | Precision_macro | Recall_macro | F1_macro |
|---|---|---|---|---|
| baseline_majority | 0.776 | 0.299 | 0.400 | 0.329 |
| baseline_dow_mode | 0.459 | 0.358 | 0.363 | 0.283 |
| logreg | 0.653 | 0.334 | 0.296 | 0.284 |

---

## 4. 考察 — なぜここまで差がつくのか

### 4-1. R² が負である件は計算上正しい

R² = 1 − SS_res / SS_tot。SS_tot は **検証 fold 内平均** からの分散なので、訓練 fold 平均と検証 fold 平均がズレる（非定常）と、「学習平均で予測」する単純ベースラインは検証平均から系統的に外れて R² 大幅マイナスになる。bento の `baseline_mean` R² = −3.43 がまさにそれ。

### 4-2. bento は強いトレンドを持つ

TimeSeriesSplit の各 fold での平均：

| fold | train mean | test mean | 差 |
|---|---|---|---|
| 0 | 130.2 | 108.3 | −21.9 |
| 1 | 119.7 | 84.6 | −35.1 |
| 2 | 108.3 | 73.4 | −34.9 |
| 3 | 99.8 | 61.0 | −38.8 |
| 4 | 92.2 | 58.4 | −33.8 |

開催期間中に売上が逓減し続けている（SIGNATE「お弁当の需要予測」の有名な性質）。`name`（メニュー）も特徴量に入っていないため、トレンドを捉える情報源が特徴量にほぼ無い。負の R² は **手法ではなく特徴量不足のサイン**。

### 4-3. sushi（仮想）は定常 + 強い決定論シグナル

同じ統計を sushi_unified で取ると：

| fold | train mean | test mean | 差 |
|---|---|---|---|
| 0 | 49.1 | 42.9 | −6.2 |
| 1 | 46.1 | 48.6 | +2.5 |
| 2 | 46.9 | 44.5 | −2.4 |
| 3 | 46.3 | 47.6 | +1.3 |
| 4 | 46.6 | 42.9 | −3.7 |

線形 slope = −0.0045/日（3 年通算で平均の 10% 未満）。fold 間で train 分布 ≈ test 分布。さらに曜日ごとの平均に強い偏り（水曜 82.3 / 火 33.2 / 金 32.0）があり、**「曜日ダミー 1 個」だけで R² 0.6 を稼げる構造**。これは `data/generate_data.py` で曜日・天気・休日が決定論的に売上に効くよう生成されているため。

### 4-4. つまり sushi の CV は何を測っているのか

**Yes（CV の本来目的としては機能していない部分）**
fold 間で分布が変わらないため、「学習期間 → 未来期間でレジームが変わったときの頑健性」は測れない。bento のような実データ非定常性に対するストレステストにはならない。

**No（測れている部分はある）**
「曜日／天気／休日 → sales_count の関数を学習する能力」のテストとしては機能している。rf が linear を大きく上回った（R² 0.10 → 0.85）のは、水曜ピーク等の非線形性を捉えた証拠で、モデル比較としては意味がある。

### 4-5. 並列ベンチを残す価値

「合成では勝ち、実データでは負ける」が同じパイプラインで定量比較できる構成になっているため、合成ベンチの限界を可視化する装置として機能する。**G 検定のための教材としても、「CV で R² 高い ≠ 実運用で使える」という重要な落とし穴を示す具体例**になっている。

---

## 5. 含意と次の一手

1. **methodology / model_comparison ドキュメントに追記**：
   sushi の高スコアは仮想データの定常性に依存していること、bento と並べた数字こそが現実的な期待値を示すこと、を明記。
2. **sushi 生成スクリプトの拡張案**：
   `data/generate_data.py` に緩やかなトレンドや CPI 連動の非線形効果を仕込めば、CV が「非定常頑健性」を測れる課題になる（v2.0 ロードマップ候補）。
3. **bento 改善余地**：
   負の R² は特徴量不足の診断結果。トレンド項（経過日数、月次ダミー）、ラグ・移動平均、`name` のメニュー埋め込みを入れれば改善可能性がある。

---

## 6. 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/benchmark.py` | `_unify_products()` 関数追加、`--unify-products` CLI フラグ追加、出力タグを `sushi_unified` に切替 |
| `benchmarks/sushi_unified_*_20260429_124322.csv` 他 | 集約ベンチ結果（タイムスタンプ付きと latest を出力） |
