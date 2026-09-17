# MATLAB model — FL-AV problem design

A compact, dependency-free MATLAB model of the federated-learning problem
this repository solves in Python (`fedav/`): a fleet of simulated vehicles
("clients") each hold a private, non-IID data shard and jointly train one
shared linear model by exchanging only model weights with a Road-Side Unit
(RSU) that performs quality-aware **FedAvg** aggregation — no raw data ever
leaves a client.

It does not require any MATLAB toolbox and runs unmodified in
[GNU Octave](https://octave.org/) (verified against Octave 8.4).

## Files

| File | Purpose |
|---|---|
| `fedavg_simulation.m` | Main script/function. Builds the client fleet, runs `R` federated rounds of local SGD + aggregation, prints per-round loss, and saves a convergence plot. |
| `run_tests.m` | Dependency-free sanity tests (aggregation math, local SGD decreases loss, full simulation converges). |

## Running it

### MATLAB

```matlab
cd matlab
fedavg_simulation   % run the model, prints per-round loss + saves convergence.png
run_tests            % run the sanity tests
```

### GNU Octave (no MATLAB license required)

```bash
cd matlab
octave --no-gui -q --eval "fedavg_simulation"
octave --no-gui -q --eval "run_tests"
```

Expected output ends with something like:

```
round 40/40  |  global loss = 0.269684
final weight error ||w_hat - w*|| = 0.401802
convergence plot saved to matlab/convergence.png
```

and `run_tests` prints `All tests passed.` with exit code 0 (suitable for CI).

## What the model captures

- **Non-IID partitioning** — each client's feature distribution is shifted
  and scaled differently (`make_clients`), and label noise varies per
  client, mirroring heterogeneous sensor quality across a vehicle fleet.
- **Client selection** — only a fraction `C` of clients participate each
  round (`select_clients`), like the PRD's connectivity-aware selection.
- **Local training** — each selected client runs several epochs of local
  gradient descent on its own shard before reporting weights
  (`local_sgd_update`).
- **Quality-aware aggregation** — the RSU averages client weights weighted
  by a quality score (`client_quality`, `aggregate_models`) instead of a
  plain average, matching the `q_k` weighting in the PRD.
- **Convergence tracking** — global MSE loss is recomputed after every
  round and plotted (`plot_convergence`).

Tune the experiment by editing `default_config()` at the top of
`fedavg_simulation.m` (number of clients, rounds, local epochs, learning
rate, selection fraction, non-IID skew, and whether quality-aware
aggregation is enabled).
