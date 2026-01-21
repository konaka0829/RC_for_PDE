# サンプル

## Kuramoto–Sivashinsky (KS) の例

KS のデータ生成と時系列予測のデモを実行します:

```bash
python examples/kuramoto-sivashinsky.py \
  --ks-steps 1000 \
  --train-length 700 \
  --predict-length 100 \
  --ks-n 64 \
  --reservoir-size 512
```

### KS 固有の実装ポイント

KS のユーティリティとリザバー実装は、MATLAB 参照コードの挙動に合わせています:

- **ブロック構造の入力重み**: リザバーを等分割したブロックごとに 1 つの入力変数が対応します。
  計算は `repeat_interleave` 相当で効率化しています。
- **偶数インデックスの二乗**: MATLAB の偶数インデックス (2,4,6,...) を二乗する特徴量は、
  Python の `1,3,5,...` を二乗する処理に対応します。
- **疎行列のスペクトル半径調整**: MATLAB `sprand` 相当の U(0,1) サンプルで疎行列を生成し、
  パワーイテレーションでスペクトル半径を推定して所望の値にスケーリングします。

### Fig.2 風プロットの再現設定

`torchesn.ks.plots` のヘルパーで PRL Fig.2 風の可視化を再現できます:

```python
import matplotlib.pyplot as plt

from torchesn.ks import reproduce_prl_figure2

fig, (actual, pred, err), (t_axis, x_axis) = reproduce_prl_figure2(
    seed_data=0,      # 初期条件の乱数種（固定したいなら指定）
    seed_A=0,         # リザバーAの乱数種（固定したいなら指定）
    approx_reservoir_size=10000,  # 論文Fig.2は Dr=5000
    sigma=1.0,        # 論文本文の代表値
    train_length=70000,
    predict_length=1000,  # 横軸を ~12 Lyapunov time に合わせやすい
    xlim=(0.0, 12.0),
)

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"fig2_like.{ext}", dpi=200)
plt.show()
```

## KS 並列リザバーの追加機能

KS の並列リザバー実装およびベンチマーク・図再現のための関数が追加されました。
主な追加項目は次の通りです:

- `KSParallelReservoir` / `KSParallelParams`: 空間を分割した並列リザバー。
- `benchmark_parallel_reservoir`: PRL スタイルの RMSE 評価ループ。
- `ks_solve_etdrk4_forced`, `generate_ks_dataset`: 強制項付き KS ソルバとデータ生成。
- `reproduce_prl_fig4/fig5a/fig5b/fig6`: PRL 図の再現ヘルパー。

### 簡単な実行例 (データ生成 → 学習 → 予測)

```python
import torch

from torchesn.ks import (
    KSParallelParams,
    KSParallelReservoir,
    generate_ks_dataset,
)

u = generate_ks_dataset(
    Q=64,
    L=200.0,
    mu=0.01,
    wavelength=100.0,
    dt=0.25,
    n_steps=5000,
    burn_in=100,
    seed=0,
    device="cpu",
)

params = KSParallelParams(
    Q=64,
    g=8,
    locality=2,
    approx_reservoir_size=1000,
    radius=0.6,
    degree=3.0,
    beta=1e-4,
    sigma=1.0,
    discard_length=100,
    train_length=1000,
    predict_length=200,
    jobid=1,
)

model = KSParallelReservoir(params, device="cpu", dtype=torch.float64)
model.fit(u[:, : params.discard_length + params.train_length])

pred = model.predict_one_interval(
    u,
    warmup_start=params.discard_length + params.train_length - 10,
    sync_length=10,
    predict_length=200,
)
print(pred.shape)  # [Q, predict_length]
```

### ベンチマーク実行例 (RMSE 曲線)

```python
from torchesn.ks import benchmark_parallel_reservoir

bench = benchmark_parallel_reservoir(
    u,
    params=params,
    tau=200,
    K=3,
    epsilon=10,
    num_trials=2,
)
print(bench["rmse_mean"].shape)  # [tau]
```

### PRL 図再現例

```python
from torchesn.ks import reproduce_prl_fig4

fig, outputs = reproduce_prl_fig4()
fig.savefig("prl_fig4_like.png", dpi=150)
```
