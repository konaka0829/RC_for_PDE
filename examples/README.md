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
