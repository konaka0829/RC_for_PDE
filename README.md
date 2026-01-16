# RC_for_PDE

本リポジトリは、PDE（偏微分方程式）に対するリザバーコンピューティング（RC）実験を行うためのコード群です。  
Pathak et al. (PRL 2018) に基づく Parallel ESN、Kuramoto–Sivashinsky (KS) 方程式のデータ生成、  
および評価プロトコル（RMSE 曲線）を再現するための CLI などを含みます。

---

## 目次

1. [ディレクトリ構成](#ディレクトリ構成)
2. [主要コンポーネントの概要](#主要コンポーネントの概要)
3. [KSデータ生成](#ksデータ生成)
4. [PRL 2018 図の再現](#prl-2018-図の再現)
5. [KS評価（RMSE曲線）](#ks評価rmse曲線)
6. [テスト実行](#テスト実行)

---

## ディレクトリ構成

```
torchesn/
  nn/
    echo_state_network.py         # ESN実装（offline readout、step、autoregressive）
    parallel_echo_state_network.py# ParallelESN 実装（PRL 2018）
    reservoir.py                  # Reservoir 本体
  utils/
    kuramoto_sivashinsky.py       # KS方程式シミュレータ（ETDRK4）
    datasets_ks.py                # KSデータのキャッシュ生成・読み込み
    utilities.py                  # 補助関数（washout 等）
examples/
  prl2018_figures.py              # Fig.2/4/5/6 再現スクリプト
  eval_ks_rmse.py                 # KS評価（RMSE曲線）スクリプト
  make_ks_dataset.py              # KSデータセット生成CLI
tests/                             # pytestテスト群
```

---

## 主要コンポーネントの概要

### ESN / Reservoir
- **`torchesn/nn/echo_state_network.py`**  
  ESN 本体。`readout_features=linear_and_square` で二次特徴を追加可能。  
  `step()` で 1 ステップ更新、`predict_autoregressive()` で自律予測が可能です。

- **`torchesn/nn/reservoir.py`**  
  リザバーの重み初期化、スペクトル半径調整（大規模時は power iteration を使用）。

### ParallelESN
- **`torchesn/nn/parallel_echo_state_network.py`**  
  PRL 2018 の Parallel Reservoirs を実装。  
  1D 周期境界の空間格子を分割し、各リザーバが局所入力を受け取る構成です。

---

## KSデータ生成

### KS方程式シミュレータ
ファイル: `torchesn/utils/kuramoto_sivashinsky.py`

KS方程式（周期境界・外力あり）:
```
u_t = -u u_x - u_xx - u_xxxx + μ cos(2π x / λ)
```

- ETDRK4 を用いた安定な時間積分
- `simulate_ks(...)` で `shape=(n_steps, Q)` の時系列を返す
- `burn_in` で初期トランジェントを除外可能

### KSデータセットのキャッシュ
ファイル: `torchesn/utils/datasets_ks.py`

`generate_or_load_ks_dataset(...)` で以下を行います：
- 指定パラメータに一致する `.npz` があればロード
- 無ければ新規生成して保存

### 例：KSデータ生成

```bash
python examples/make_ks_dataset.py \
  --output examples/datasets/ks_dataset.npz \
  --L 200 --Q 512 --dt 0.25 --mu 0.01 --lam 100 \
  --total-steps 6000 --burn-in 1000 --seed 0
```

---

## PRL 2018 図の再現

スクリプト: **`examples/prl2018_figures.py`**

### Fig.2 (single reservoir 相当)
```bash
python examples/prl2018_figures.py fig2_single --quick --out-dir /tmp/prl2018_figs
```

### Fig.4 (parallel reservoirs)
```bash
python examples/prl2018_figures.py fig4_parallel --quick --out-dir /tmp/prl2018_figs
```

### Fig.5 (スケーリング)
```bash
python examples/prl2018_figures.py fig5_scaling --quick --out-dir /tmp/prl2018_figs
```

### Fig.6 (shared weights)
```bash
python examples/prl2018_figures.py fig6_shared_weights --quick --out-dir /tmp/prl2018_figs
```

**ポイント**
- `--quick` はCI用の短時間設定です。  
  論文再現を狙う場合は `--paper-defaults` を利用してください。
- 出力は PNG で保存されます。

---

## KS評価（RMSE曲線）

スクリプト: **`examples/eval_ks_rmse.py`**

### quick モード
```bash
python examples/eval_ks_rmse.py --quick --out-dir /tmp/ks_eval
```

### 出力
- `rmse_curve.png` : RMSE 曲線
- `rmse_curve.json` : パラメータと RMSE 配列

### 代表パラメータ例（論文規模）
```bash
python examples/eval_ks_rmse.py \
  --L 200 --Q 512 --dt 0.25 --mu 0.01 --lam 100 \
  --T-train 70000 --K 30 --tau 1000 --epsilon 10 --n-trials 10 \
  --g 64 --l 6 --hidden-size 5000 --spectral-radius 0.6 \
  --lambda-reg 1e-4 --readout-features linear_and_square
```

---

## テスト実行

```bash
python -m pytest -q
```

CI向けの quick テストでは以下が確認されます：
- PRL 2018 図生成が完走する
- RMSE評価が完走し、PNG/JSONが生成される
