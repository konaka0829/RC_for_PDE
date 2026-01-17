# RC_for_PDE

PDE（偏微分方程式）向けのリザバー計算（Reservoir Computing）関連コードと実験用ユーティリティを集めたリポジトリです。

## バージョン
- **Python 3 系（バージョン 3）**を採用しています。

## ディレクトリ構成
- `torchesn/`
  - コア実装（ESN/Reservoir など）とユーティリティ。
  - `torchesn/nn/` にニューラルネットワーク関連実装。
  - `torchesn/utils/` にデータ前処理やシミュレータ等。
- `tests/`
  - ユニットテスト群。
- `examples/`
  - サンプルスクリプト（例: `mackey-glass.py`, `mnist.py`）。
- `reports/`
  - 実験結果の記録など。
- `requirements.txt` / `testing_requirements.txt`
  - 実行・テスト用依存関係。

## セットアップ
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r testing_requirements.txt
```

## 実行方法
### 例: Mackey-Glass サンプル
```bash
python examples/mackey-glass.py
```

### KS シミュレーション（ユーティリティ）
```python
from torchesn.utils import simulate_ks

u = simulate_ks(L=22.0, Q=64, dt=0.001, mu=-1.0, lam=0.0, n_steps=100, seed=0)
print(u.shape)  # (101, 64)
```

## テスト
```bash
pytest -q
```
