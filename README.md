# RC for PDE (Kuramoto–Sivashinsky)

このリポジトリは、Kuramoto–Sivashinsky (KS) 方程式に対する
リザバーコンピューティング実装（単一リザバー／並列リザバー）を中心に構成されています。
KS データ生成やベンチマークのユーティリティが `torchesn/ks` に集約されています。

## 主要な構成

- `torchesn/ks`: KS 方程式向けのソルバ、単一／並列リザバー、ベンチマーク、図再現のヘルパー。
- `examples/`: 実行可能なサンプルスクリプト。

## サンプル実行

KS のデータ生成と時系列予測のデモ:

```bash
python examples/kuramoto-sivashinsky.py \
  --ks-steps 1000 \
  --train-length 700 \
  --predict-length 100 \
  --ks-n 64 \
  --reservoir-size 512
```

詳細は `examples/README.md` を参照してください。

## KS 向け主要 API

`torchesn.ks` には以下の主要 API があります:

- `KSBasicSingleReservoir`, `KSReservoirParams`: 単一リザバー実装。
- `KSParallelReservoir`, `KSParallelParams`: 空間を分割した並列リザバー実装。
- `generate_ks_dataset`, `ks_solve_etdrk4_forced`, `kursiv_solve_etdrk4`: データ生成とソルバ。
- `benchmark_parallel_reservoir`, `rmse_over_space`: ベンチマーク補助。
- `reproduce_prl_figure2`, `reproduce_prl_fig4/fig5a/fig5b/fig6`: PRL 図再現ヘルパー。

これらは Python から直接呼び出して利用できます。
