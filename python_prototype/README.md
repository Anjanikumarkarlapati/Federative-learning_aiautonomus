# Python prototype

A self-contained, numpy-only federated-learning demo for collaborative autonomous-vehicle
(AV) training. For the full PyTorch framework see the repository root (`fedav/`).

```
pip install numpy            # matplotlib optional, only for --plot
python python_prototype/fl_prototype.py
python python_prototype/fl_prototype.py --clients 40 --rounds 50 --repeats 10 --plot
python python_prototype/fl_prototype.py --quiet
```

## The idea
N vehicles train one shared model; raw data never leaves a vehicle. Each round the
Road-Side Unit (RSU) does:

1. **Select** a fraction `C` of vehicles.
2. **Train** locally: each vehicle runs SGD on its private data.
3. **Upload** only the delta `dW_k = w_k - w` (optionally clipped + noised, top-k sparsified).
4. **Aggregate** `w <- w + sum_k a_k dW_k`, with `a_k = q_k n_k / sum_j q_j n_j`
   (`n_k` = samples, `q_k = 1/(1 + local loss)` = quality score).

## What is simulated
| Effect | How |
|---|---|
| Label skew | per-vehicle class mix `~ Dirichlet(alpha)` (`--alpha`, smaller = more skew) |
| Quantity skew | log-normal number of samples per vehicle |
| Feature shift | per-vehicle offset (region / weather / time of day) |
| Sensor quality | per-vehicle noise level |
| Connectivity | per-vehicle uptime; uploads from unreliable vehicles are dropped |
| Privacy | per-update L2 clipping + Gaussian noise (illustrative; no formal (eps, delta) accounting) |
| Bandwidth | top-k sparsification with error feedback; bytes counted (fp16 values + int32 indices) |

## Arms compared
`centralized` (upper bound, pooled data) · `fedavg` · `+selection` · `+quality` ·
`proposed` (both) · `proposed+topk` · `proposed+dp`.

## Output
Console: fleet table, per-round log of the proposed method, and a results table
(accuracy mean ± std over `--repeats` seeds, worst-vehicle accuracy, rounds to 90% of final
accuracy, uploaded KB, dropped uploads). Files in `python_prototype/results/`:
`rounds.csv` (per round, first seed), `summary.csv`, and `accuracy.png` with `--plot`.

## Reading the results honestly
The toy model is small and the gaps between fedavg and proposed are often within one
standard deviation across seeds; use `--repeats 10` or more before concluding anything.
The top-k saving is modest because index overhead dominates a ~100-weight model.
