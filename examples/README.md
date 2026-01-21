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

## KS 並列リザバーの例

並列リザバーでの学習と RMSE ベンチマークを実行します:

```bash
python examples/ks-parallel.py \
  --n-steps 5000 \
  --train-length 1000 \
  --predict-length 200 \
  --approx-reservoir-size 1000
```
