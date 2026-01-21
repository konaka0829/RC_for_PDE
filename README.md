# RC_for_PDE

このリポジトリは、Kuramoto–Sivashinsky (KS) 方程式のデータセット生成と、
シングル/並列リザーバコンピューティングによる予測を行うための実装です。

## ディレクトリ構成

```
src/rc_for_pde/
  cli/            # 予測実行用CLI
  ks/             # KSソルバ（ETDRK4）とデータ生成
  reservoirs/     # シングル/並列リザーバ実装
  benchmarks/     # ベンチマーク用ユーティリティ
```

## セットアップ

```bash
pip install -r requirements.txt
```

主な依存関係:

- torch
- numpy
- matplotlib

## 予測の実行

`src` 配下を読み込むため、`PYTHONPATH=src` を設定して実行します。

### シングルリザーバ

```bash
PYTHONPATH=src python -m rc_for_pde.cli.predict \
  --mode single \
  --ks-n 64 \
  --ks-d 22 \
  --ks-steps 100000 \
  --train-length 70000 \
  --predict-length 1000 \
  --plot --plot-path outputs/single.png \
  --time-axis scaled
```

### 並列リザーバ

```bash
PYTHONPATH=src python -m rc_for_pde.cli.predict \
  --mode parallel \
  --Q 512 --L 200 --mu 0.01 --wavelength 100 \
  --discard-length 1000 --parallel-train-length 79000 \
  --parallel-predict-length 2999 --g 64 --locality 6 \
  --plot --plot-path outputs/parallel.png \
  --time-axis discrete
```

## 主なオプション

- `--mode` : `single` または `parallel`
- `--time-axis` : `scaled`（リアプノフ指数で正規化）または `discrete`（離散時間）
- `--plot` / `--plot-path` : 予測結果のプロット出力

詳細は `PYTHONPATH=src python -m rc_for_pde.cli.predict --help` を参照してください。
