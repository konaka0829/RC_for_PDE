# KS Single Reservoir ESN デモ (日本語)

このサンプルは、Kuramoto–Sivashinsky (KS) 方程式の 1 ステップ先予測を
単一リザーバの ESN (Echo State Network) で行う最小構成のデモです。
`simulate_ks_etdrk4` を使って KS データを生成し、
**オフライン学習 (ridge 回帰, Cholesky)** で readout を学習した後、
**自己回帰 (closed-loop) 予測**を実行します。

## 主な仕様

- **学習タスク**: 1 ステップ先予測 `u(t) -> u(t+dt)`
- **学習**: `readout_training="cholesky"` を用いた ridge 回帰
- **特徴変換**: `feature_transform="square_even"` を学習/推論の両方に適用
- **MATLAB 寄り設定**:
  - block Win (`--input-init block`)
  - degree=3 相当の疎な再帰結合 (`density = degree / hidden_size`)
  - bias 無し (`--reservoir-bias`/`--readout-bias` を使わない)
- **検証**: 予測区間の RMSE を算出し、プロットとして保存
- **出力**:
  - RMSE 時系列プロット (`pdf/png/svg`)
  - 予測結果のヒートマップ (`pdf/png/svg`)

## 使い方

### 1) 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2) 実行

```bash
python examples/ks_single_reservoir.py
```

実行後、`outputs/` 配下に以下が生成されます。

- `ks_single_reservoir_rmse.png`
- `ks_single_reservoir_rmse.pdf`
- `ks_single_reservoir_rmse.svg`
- `ks_single_reservoir_prediction_heatmap.png`
- `ks_single_reservoir_prediction_heatmap.pdf`
- `ks_single_reservoir_prediction_heatmap.svg`

### 3) 主なオプション

```bash
python examples/ks_single_reservoir.py \
  --n-grid 64 \
  --dt 0.25 \
  --domain-length 22.0 \
  --mu 0.0 \
  --train-steps 2000 \
  --pred-steps 500 \
  --warmup-steps 200 \
  --hidden-size 500 \
  --spectral-radius 0.9 \
  --leaking-rate 1.0 \
  --lambda-reg 1e-4 \
  --output-dir outputs
```

- `--no-plots` を指定すると、RMSE/ヒートマップ出力をスキップできます。
- `--input-init block` の場合は `hidden_size` が `n_grid` の倍数になるように
  自動調整されます。

### 4) MATLAB 寄りの推奨例 (軽め)

```bash
python examples/ks_single_reservoir.py \
  --n-grid 64 \
  --train-steps 2000 \
  --pred-steps 500 \
  --warmup-steps 200 \
  --hidden-size 512 \
  --spectral-radius 0.6 \
  --degree 3 \
  --win-sigma 0.5 \
  --lambda-reg 1e-3 \
  --no-plots
```

## 予測の仕組み (簡単な説明)

1. KS シミュレーションで `u(t)` を生成します。
2. `u(t) -> u(t+dt)` の教師ありデータを作成します。
3. ESN の readout を ridge 回帰で学習します。
4. warmup 区間は真値入力で隠れ状態を整え、以降は **予測出力を次の入力**
   として自己回帰で予測します。
5. 予測区間の RMSE を計算し、結果を保存します。

## 自分の環境での実行方法

1. **Python 3.10+ を推奨**
2. リポジトリを取得し、依存関係をインストールします。
3. 上記のコマンドで `ks_single_reservoir.py` を実行します。
4. 予測結果のプロットが `outputs/` に出力されます。

もし計算が重い場合は、以下のように小さな設定で試すと軽くなります。

```bash
python examples/ks_single_reservoir.py \
  --n-grid 16 \
  --train-steps 300 \
  --pred-steps 50 \
  --warmup-steps 50 \
  --hidden-size 200 \
  --lambda-reg 1e-3
```
