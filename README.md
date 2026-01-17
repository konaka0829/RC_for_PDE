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
  --total-steps 90000 --train-steps 70000 --seed 0
```

補足:
- 論文では 70,000 ステップを学習に使用しています。上の例では total_steps=90,000 として test 用に 20,000 ステップを確保しています。
- 論文設定は非常に重いので、まずは軽量デモで動作確認してください。

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
