# Streamlit Cloud デプロイ手順

> 📝 本ドキュメントは Claude Code が用意した運用手順書です。
> M5 時点ではローカル動作のみ確認済み、Streamlit Cloud へのデプロイはユーザーが本手順に沿って実施します。

## 1. 前提

- GitHub にリポジトリが push 済み（M5の最終commit `v1.0.0-m5` が main ブランチに上がっている）
- Streamlit Community Cloud のアカウント（無料、GitHub アカウントでサインアップ可）
  - https://share.streamlit.io/

## 2. デプロイ手順

### Step 1：Streamlit Cloud にサインイン

1. https://share.streamlit.io/ にアクセス
2. 「Continue with GitHub」で GitHub アカウントを連携
3. 初回はリポジトリへのアクセス権限を求められるので承認

### Step 2：新しいアプリを作成

1. ダッシュボードで「New app」をクリック
2. 以下を入力：
   - **Repository**：`<your-account>/sushi-predictor`
   - **Branch**：`main`
   - **Main file path**：`app.py`
   - **App URL**（任意）：`sushi-predictor.streamlit.app` のようなサブドメインが使える

### Step 3：デプロイ実行

1. 「Deploy!」をクリック
2. 初回ビルドは 2〜5 分かかる（依存パッケージのインストール含む）
3. 完了するとアプリ画面が表示される

## 3. 既知の注意点

### 3.1 学習済みモデルは含まれない

`.gitignore` で `models/*.pkl` を除外しているため、デプロイ先には学習済みモデルが存在しません。**初回起動時に Streamlit が自動学習**します（`app.py` の `_cached_models()` 関数で `train_all()` が実行される設計）。

- 初回ロード時間：4 モデル × 4 商品 = 16 モデル × CV 5-fold ≒ 30〜60 秒
- 2 回目以降は `@st.cache_resource` でメモリにキャッシュ

> **TIP**：「初回が遅い」と感じる場合は、`models/*.pkl` を git LFS や Releases にアップロードして `app.py` の起動時にダウンロードする運用に変更可能。

### 3.2 メモリ上限

Streamlit Community Cloud の無料枠は **1 GB RAM、1 CPU**。本プロジェクトは scikit-learn + LightGBM 程度なので問題ない想定だが、もし `Resource limits exceeded` が出た場合：

- LightGBM の `n_estimators` を減らす（200 → 100）
- ランダムフォレストの `n_estimators` を減らす
- Streamlit の `@st.cache_resource` を最大限活用

### 3.3 リポジトリの可視性

**Public リポジトリ**前提です。Private にするには Streamlit Cloud の Teams プラン（有料）が必要。

## 4. デプロイ後にやること

1. **README.md にライブデモ URL を追記**
   ```markdown
   ## 2. デモ
   👉 ライブデモ：https://sushi-predictor.streamlit.app/
   ```
2. アプリを実際に開いて、4 モデルすべてで予測が走ることを確認
3. スクリーンショットを撮影し `docs/images/` 配下に配置（README からリンク）
4. GitHub プロフィールの Pinned Repositories に `sushi-predictor` を追加
5. ポートフォリオ用途で SNS や履歴書に URL を掲載

## 5. トラブルシューティング

### 5.1 ビルドエラー：パッケージが見つからない

`requirements.txt` の依存にバージョン上限がない場合、Streamlit Cloud 側で最新版を取りに行って互換性問題が発生することがある。固定したい場合は：

```text
# requirements.txt
streamlit==1.56.0
pandas==3.0.2
scikit-learn==1.8.0
lightgbm==4.6.0
```

のようにバージョンを `==` で固定。

### 5.2 起動が遅い / タイムアウト

初回学習が長引いてタイムアウトする場合は、ローカルで `python -m src.trainer` を実行し、`models/*.pkl` を Git LFS でリポジトリに含める運用に切り替える。

### 5.3 文字化け

`.streamlit/config.toml` で日本語フォントが指定されていないため、ブラウザの既定フォントに依存します。問題があれば Streamlit の `[theme]` セクションで `font` を指定。

## 6. アプリの停止・削除

- ダッシュボードからアプリを選択 → 「Settings」→「Delete」
- 一時停止は「Reboot」または「Stop」（再起動すると初回学習が走る点に注意）

---

**最終更新**：2026-04-28
**前提のリポジトリ状態**：M5（タグ `v1.0.0-m5`）
