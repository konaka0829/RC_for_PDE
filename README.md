# Kuramoto–Sivashinsky (KS) 並列リザーバ計算

このリポジトリは、Kuramoto–Sivashinsky (KS) 方程式データの生成と、Pathak et al. (PRL 2018) の並列リザーバ計算 (PRC) を再現するための実験フローを提供します。

- KS データセット生成（ETDRK4）
- 単一/並列リザーバ予測（teacher forcing → 自律予測）
- 予測区間を複数回評価し RMSE を集計

## インストール

```bash
pip install -r requirements.txt
```

## KS データ生成

### 軽量デモ

```bash
python examples/ks/generate_ks_dataset.py \
  --out /tmp/ks_demo.npz \
  --L 50 --Q 32 --dt 0.25 --mu 0.01 --lambda 25 \
  --total-steps 4000 --train-steps 3000 --seed 0
```

### 論文設定（Pathak et al. PRL 2018）

```bash
python examples/ks/generate_ks_dataset.py \
  --out /tmp/ks_paper.npz \
  --L 200 --Q 512 --dt 0.25 --mu 0.01 --lambda 100 \
  --total-steps 100010 --train-steps 70000 --seed 0
```

補足:
- 論文では 70,000 ステップを学習に使用しています。`--paper` 実行には test_steps>=30,010 が必要なため、上の例では total_steps=100,010 としています。
- 論文設定は非常に重いので、まずは軽量デモで動作確認してください。

## 論文設定を現実的に回すコツ（高速化・省メモリ）

論文設定スケール（例: g=64, Dr≈5000, train_length=70000, K=30, predict_length=1000）では学習時のメモリ・計算負荷が非常に大きくなります。以下のオプションを組み合わせて、Colab/GPU 環境でも現実的に回せるようにしてください。

### dtype（推奨: float32）

- **RC 学習は `--dtype float32` を推奨**します。Dr×Dr の統計行列や演算のメモリ/速度が改善します。
- データ生成（`generate_ks_dataset.py`）は NumPy FFT の CPU 実行であり dtype は任意です。**データが float64 でも、実行スクリプト側で `--dtype float32` により torch テンソルへキャスト可能**です。

### device（推奨: cuda）

- **`--device cuda` を推奨**します（PyTorch が CUDA 対応である必要があります）。
- もし環境によって `torch.sparse.mm` が CUDA で未対応/エラーになる場合は `--device cpu` に戻してください。
- 目安: **まず CPU で動作確認 → CUDA に切り替え**ると安心です。

### chunk_size（省メモリ・高速化トレードオフ）

学習時に状態行列 X を全保持せず、チャンク単位で XXT/YXT を蓄積して ridge を解きます。**OOM が出たら chunk_size を下げてください**。

- 余裕がある GPU: `--chunk-size 2048`
- 標準: `--chunk-size 1024`
- メモリ不安: `--chunk-size 256`〜`512`

### 実用的注意

- `--paper` の設定では、必要な test 長は以下です:
  - `required_test = (K-1)*stride + (sync_length + predict_length)`
  - 論文 preset では `required_test = 30,010` のため、**`total_steps >= train_steps + 30,010`** が必要です。
- 生成データのサイズ目安: Q=512, total_steps≈100k なら **float64 は数百 MB 級、float32 なら概ね半分**。
- `--share-weights` は **mu=0（並進対称）でのみ学習を大幅に削減できます**が、**mu=0.01（論文 Fig.4 設定）では推奨しません**。

### 具体的コマンド例（必須）

#### 論文設定データ生成

```bash
python examples/ks/generate_ks_dataset.py \
  --out /tmp/ks_paper.npz \
  --L 200 --Q 512 --dt 0.25 --mu 0.01 --lambda 100 \
  --total-steps 100010 --train-steps 70000 --seed 0
```

#### 論文設定実行（GPU 推奨）

```bash
python examples/ks/run_parallel_rc_ks.py \
  --data /tmp/ks_paper.npz \
  --paper --device cuda --dtype float32 --chunk-size 1024
```

#### 論文設定実行（CPU でも回す場合）

```bash
python examples/ks/run_parallel_rc_ks.py \
  --data /tmp/ks_paper.npz \
  --paper --device cpu --dtype float32 --chunk-size 512
```

#### まずは軽量デモで確認

```bash
python examples/ks/run_parallel_rc_ks.py \
  --data /tmp/ks_demo.npz \
  --g 4 --l 1 \
  --reservoir-size-approx 200 \
  --degree 3 --spectral-radius 0.6 --sigma 1.0 --beta 1e-4 \
  --train-length 3000 --train-discard 0 \
  --predict-length 100 --sync-length 10 \
  --num-intervals 3 --interval-stride 100 \
  --seed 0 --dtype float32 --device cpu --chunk-size 64
```

## 並列リザーバ学習 + 予測

### 軽量デモ

```bash
python examples/ks/run_parallel_rc_ks.py \
  --data /tmp/ks_demo.npz \
  --g 4 --l 1 \
  --reservoir-size-approx 200 \
  --degree 3 --spectral-radius 0.6 --sigma 1.0 --beta 1e-4 \
  --train-length 3000 --train-discard 0 \
  --predict-length 100 --sync-length 10 \
  --num-intervals 3 --interval-stride 100 \
  --seed 0
```

### 論文設定（Pathak et al. PRL 2018）

```bash
python examples/ks/run_parallel_rc_ks.py \
  --data /tmp/ks_paper.npz \
  --paper
```

`--paper` フラグの設定内容:
- **KS**: L=200, Q=512, mu=0.01, lambda=100, dt=0.25（データセットが一致している前提）
- **PRC**: g=64, l=6, reservoir_size≈5000, degree=3, spectral_radius=0.6, sigma=1.0
- **Training**: 70,000 steps（train_length）、discard なし
- **Prediction**: K=30 区間、predict_length=1000、sync_length=10、stride=1000
- **beta**: 1e-4

※ 論文設定は計算負荷が高いため、最初は軽量デモで実行を推奨します。

## RMSE プロット

平均 RMSE 曲線を画像として保存できます（png/pdf/svg を同時出力）。

```bash
python examples/ks/run_parallel_rc_ks.py \
  --data /tmp/ks_demo.npz \
  --plot-out /tmp/ks_rmse.png
```

## テスト

```bash
pytest -q
```
