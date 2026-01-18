# RC_for_PDE

Kuramoto–Sivashinsky (KS) 系列のデータ生成と、リザバー計算 (Reservoir Computing) による学習・予測を行うための Python/PyTorch 実装です。
MATLAB 版の挙動を忠実に再現することを重視しています。

## 構成

- `ks_basic_single_reservoir_torch/`
  - KS ソルバ (`kursiv_solve`) とリザバー学習/予測の実装。
- `examples/`
  - MATLAB の `ks.m` に相当する end-to-end デモスクリプト。
- `tests/`
  - KS ソルバ、リザバー学習、ストリーミング学習、end-to-end のテスト群。

## 主要機能

- **KS ソルバ**: `torch.fft` を用いた Kuramoto–Sivashinsky 方程式の時間発展。
- **リザバー学習**:
  - MATLAB 版 `train.m` の挙動を再現。
  - 偶数行二乗の特徴量拡張、スペクトル半径調整などを実装。
- **ストリーミング学習**:
  - `states` 行列を全保持せず、`S = XX^T` と `D = YX^T` をブロックで蓄積。
  - MATLAB と数学的に同値の `wout` を得る。
- **デモ & テスト**:
  - `examples/ks_basic_single_reservoir_demo.py` で end-to-end 実行。
  - `pytest -q` で一括テスト。

## セットアップ

```bash
pip install -r requirements.txt
pip install -r testing_requirements.txt
```

## 実行例

```bash
python examples/ks_basic_single_reservoir_demo.py
```

## テスト実行

```bash
pytest -q
```
