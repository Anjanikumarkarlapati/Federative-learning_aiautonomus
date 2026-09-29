"""Federated learning prototype for collaborative autonomous-vehicle (AV) training.

WHAT THIS DEMONSTRATES
----------------------
A fleet of N vehicles jointly trains ONE shared model without any vehicle sharing
raw sensor data. Each round:

    1. SELECT   the Road-Side Unit (RSU) picks a fraction C of the fleet.
    2. TRAIN    each chosen vehicle runs a few epochs of SGD on its private data.
    3. UPLOAD   the vehicle sends only the weight delta  dW_k = w_k - w  (optionally
                clipped + noised for privacy, and top-k sparsified to save bandwidth).
    4. AGGREGATE the RSU forms   w <- w + sum_k a_k dW_k ,   a_k = q_k n_k / sum_j q_j n_j
                where n_k = #samples and q_k = quality score of vehicle k.

The "model" is a softmax classifier on synthetic sensor feature vectors. It is a
stand-in for the CNN in the full framework (`fedav/`), chosen so that the whole
federated pipeline stays readable and runs in seconds with numpy only.

WHY THE DATA IS HARD (non-IID)
------------------------------
    label skew      p_k ~ Dirichlet(alpha)         vehicles see different class mixes
    quantity skew   n_k ~ LogNormal                vehicles hold very different amounts
    feature shift   per-vehicle offset             region / weather / time of day
    sensor noise    per-vehicle noise level        degraded sensors give worse data
    connectivity    per-vehicle uptime in [.2, 1]  unstable V2I links drop uploads

EXPERIMENTS (arms)
------------------
    centralized     upper bound: all data pooled (violates privacy, for reference only)
    fedavg          random selection, size-weighted average (McMahan et al.)
    proposed        connectivity-aware selection + quality-aware aggregation
  ablation:
    +selection      only the connectivity-aware selection
    +quality        only the quality-aware aggregation
  extensions of `proposed`:
    +topk           top-k sparsified uploads with error feedback (bandwidth saving)
    +dp             per-update L2 clipping + Gaussian noise (privacy)

USAGE
-----
    python python_prototype/fl_prototype.py                       # default demo
    python python_prototype/fl_prototype.py --clients 40 --rounds 50 --repeats 5
    python python_prototype/fl_prototype.py --quiet               # summary only
    python python_prototype/fl_prototype.py --plot                # needs matplotlib
"""
import argparse
import csv
import os

import numpy as np


# =============================================================== 1. DATA ====
def make_fleet(rng, n_clients, n_classes, dim, alpha, mean_samples):
    """Create the vehicle fleet with private non-IID datasets and a global test set.

    Returns (clients, (X_test, y_test)). Each client is a dict with keys
    X, y (private data), noise (sensor noise std), uptime (link reliability).
    """
    # Class prototypes in feature space. Small spread => overlapping classes => hard task.
    means = rng.normal(0, 0.6, (n_classes, dim))
    clients = []
    for _ in range(n_clients):
        n = max(20, int(mean_samples * rng.lognormal(0, 0.4)))      # quantity skew
        p = rng.dirichlet(alpha * np.ones(n_classes))              # label skew
        y = rng.choice(n_classes, n, p=p)
        noise = rng.uniform(0.8, 2.5)                              # sensor quality
        shift = rng.normal(0, 0.5, dim)                            # region/weather shift
        X = means[y] + shift + rng.normal(0, noise, (n, dim))
        clients.append(dict(X=X, y=y, noise=noise, uptime=rng.uniform(0.2, 1.0)))
    # Clean, balanced global test set (what the deployed model will face).
    yt = rng.integers(0, n_classes, 1000)
    Xt = means[yt] + rng.normal(0, 1.0, (1000, dim))
    return clients, (Xt, yt)


def describe_fleet(clients, n_classes):
    """Print a per-vehicle table so the heterogeneity is visible, not hidden."""
    print("\nFLEET (private data never leaves a vehicle)")
    print(f"  {'veh':>3} {'samples':>7} {'noise':>6} {'uptime':>6}  class histogram")
    for i, c in enumerate(clients):
        hist = np.bincount(c["y"], minlength=n_classes)
        print(f"  {i:>3} {len(c['y']):>7} {c['noise']:>6.2f} {c['uptime']:>6.2f}  {hist.tolist()}")
    n = np.array([len(c["y"]) for c in clients])
    print(f"  total samples = {n.sum()}, min/median/max per vehicle = "
          f"{n.min()}/{int(np.median(n))}/{n.max()}")


# ============================================================== 2. MODEL ====
def softmax(z):
    z = z - z.max(1, keepdims=True)         # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def loss_acc(W, X, y):
    """Cross-entropy loss and accuracy of the linear-softmax model W (dim x classes)."""
    p = softmax(X @ W)
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean(), (p.argmax(1) == y).mean()


def local_train(W, X, y, rng, epochs=2, lr=0.05, batch=32):
    """Mini-batch SGD on a vehicle's private data, starting from the global weights."""
    W = W.copy()
    for _ in range(epochs):
        idx = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            b = idx[s:s + batch]
            g = softmax(X[b] @ W)
            g[np.arange(len(b)), y[b]] -= 1            # d(loss)/d(logits)
            W -= lr * X[b].T @ g / len(b)
    return W


# ====================================================== 3. FEDERATION =======
def select(rng, clients, frac, uptime_min, aware):
    """Choose ceil(frac*N) vehicles (>=2).

    aware=False : uniform random (plain FedAvg).
    aware=True  : only vehicles with uptime >= threshold are eligible, sampled with
                  probability proportional to uptime, so rounds rarely stall on
                  vehicles that are about to lose their V2I link.
    """
    k = max(2, int(np.ceil(frac * len(clients))))
    if not aware:
        return list(rng.choice(len(clients), k, replace=False))
    up = np.array([c["uptime"] for c in clients])
    ok = np.flatnonzero(up >= uptime_min)
    if len(ok) < k:                                   # not enough stable vehicles
        ok = np.argsort(-up)[:k]
    return list(rng.choice(ok, k, replace=False, p=up[ok] / up[ok].sum()))


def clip_l2(delta, clip):
    """Scale `delta` so that its L2 norm is at most `clip` (bounds each vehicle's influence)."""
    norm = np.linalg.norm(delta)
    return delta * min(1.0, clip / (norm + 1e-12))


def topk_sparsify(delta, residual, ratio):
    """Keep only the largest `ratio` fraction of entries (error feedback).

    What is not sent is remembered in `residual` and added to the next upload, so
    no information is permanently lost. Returns (sparse_delta, new_residual, nnz).
    """
    u = delta + residual
    k = max(1, int(np.ceil(ratio * u.size)))
    thresh = np.partition(np.abs(u).ravel(), -k)[-k]
    mask = np.abs(u) >= thresh
    sent = np.where(mask, u, 0.0)
    return sent, u - sent, int(mask.sum())


def per_vehicle_accuracy(W, clients):
    """Accuracy of the global model on each vehicle's own data (fairness view)."""
    return np.array([loss_acc(W, c["X"], c["y"])[1] for c in clients])


def run_fl(clients, test, dim, n_classes, rounds, frac, seed, *, aware_select,
           quality_agg, topk=1.0, dp_clip=None, dp_noise=0.0, epochs=2, verbose=False,
           label=""):
    """Run one federated experiment.

    Returns (final_weights, history) where each history row is
    (round, test_loss, test_acc, participants, dropped, uploaded_KB).
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((dim, n_classes))
    residuals = [np.zeros_like(W) for _ in clients]      # error-feedback memory, per vehicle
    hist = []
    for r in range(1, rounds + 1):
        sel = select(rng, clients, frac, 0.55, aware_select)
        deltas, weights, dropped, nbytes = [], [], 0, 0
        for i in sel:
            c = clients[i]
            if rng.random() > c["uptime"]:               # V2I link lost -> upload never arrives
                dropped += 1
                continue
            Wk = local_train(W, c["X"], c["y"], rng, epochs)
            d = Wk - W                                   # only the delta is transmitted
            n = len(c["y"])
            # quality score: vehicles that still fit their own data poorly (noisy /
            # off-distribution sensors) get a smaller aggregation weight.
            q = 1.0 / (1.0 + loss_acc(Wk, c["X"], c["y"])[0]) if quality_agg else 1.0
            if dp_clip is not None:
                d = clip_l2(d, dp_clip)
            if topk < 1.0:
                d, residuals[i], nnz = topk_sparsify(d, residuals[i], topk)
                nbytes += nnz * (2 + 4)                  # fp16 value + int32 index
            else:
                nbytes += d.size * 2                     # dense fp16
            deltas.append(d)
            weights.append(q * n)
        if deltas:
            a = np.array(weights) / sum(weights)         # a_k = q_k n_k / sum_j q_j n_j
            step = sum(ai * di for ai, di in zip(a, deltas))
            if dp_clip is not None and dp_noise > 0:
                # Gaussian mechanism on the average: sigma = z * clip / #participants
                step = step + rng.normal(0, dp_noise * dp_clip / len(deltas), step.shape)
            W = W + step
        l, acc = loss_acc(W, *test)
        hist.append((r, l, acc, len(deltas), dropped, nbytes / 1e3))
        if verbose:
            print(f"  [{label}] round {r:>3}  selected={len(sel)} received={len(deltas)} "
                  f"dropped={dropped}  test acc={acc:.3f} loss={l:.3f}  up={nbytes/1e3:.1f} KB")
    return W, hist


def run_central(clients, test, dim, n_classes, rounds, seed, epochs=2):
    """Upper bound: pool all data on one server and train (not privacy preserving)."""
    rng = np.random.default_rng(seed)
    X = np.vstack([c["X"] for c in clients])
    y = np.concatenate([c["y"] for c in clients])
    W = np.zeros((dim, n_classes))
    hist = []
    for r in range(1, rounds + 1):
        W = local_train(W, X, y, rng, epochs)
        l, a = loss_acc(W, *test)
        hist.append((r, l, a, len(clients), 0, 0.0))
    return W, hist


# ============================================================ 4. EXPERIMENT ==
ARMS = {  # name -> kwargs for run_fl
    "fedavg":          dict(aware_select=False, quality_agg=False),
    "+selection":      dict(aware_select=True,  quality_agg=False),
    "+quality":        dict(aware_select=False, quality_agg=True),
    "proposed":        dict(aware_select=True,  quality_agg=True),
    "proposed+topk":   dict(aware_select=True,  quality_agg=True, topk=0.25),
    "proposed+dp":     dict(aware_select=True,  quality_agg=True, dp_clip=1.0, dp_noise=0.3),
}


def run_all(a, seed, verbose):
    """Run every arm once on a freshly generated fleet; return {arm: (W, hist)}, clients."""
    rng = np.random.default_rng(seed)
    clients, test = make_fleet(rng, a.clients, a.classes, a.dim, a.alpha, 150)
    if verbose:
        describe_fleet(clients, a.classes)
        print("\nTRAINING (per-round log for the proposed method)")
    out = {"centralized": run_central(clients, test, a.dim, a.classes, a.rounds, seed)}
    for name, kw in ARMS.items():
        out[name] = run_fl(clients, test, a.dim, a.classes, a.rounds, a.frac, seed,
                           verbose=verbose and name == "proposed", label=name, **kw)
    return out, clients


def summarize(runs, clients):
    """Final-round metrics for one seed: {arm: dict}."""
    res = {}
    for name, (W, h) in runs.items():
        pv = per_vehicle_accuracy(W, clients)
        res[name] = dict(acc=h[-1][2], loss=h[-1][1], worst_veh=pv.min(),
                         mb=sum(x[5] for x in h),
                         dropped=sum(x[4] for x in h),
                         r90=next((x[0] for x in h if x[2] >= 0.9 * h[-1][2]), len(h)))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clients", type=int, default=20, help="fleet size N")
    ap.add_argument("--rounds", type=int, default=30, help="communication rounds R")
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--dim", type=int, default=16, help="feature dimension")
    ap.add_argument("--alpha", type=float, default=0.4, help="Dirichlet skew (small = more skew)")
    ap.add_argument("--frac", type=float, default=0.3, help="participation fraction C")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3, help="independent seeds to average")
    ap.add_argument("--quiet", action="store_true", help="skip fleet table and per-round log")
    ap.add_argument("--plot", action="store_true", help="save accuracy.png (needs matplotlib)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()

    print(f"FL-AV prototype: N={a.clients} vehicles, R={a.rounds} rounds, C={a.frac}, "
          f"Dirichlet alpha={a.alpha}, repeats={a.repeats}")
    allres, first_runs = [], None
    for k in range(a.repeats):
        runs, clients = run_all(a, a.seed + k, verbose=(not a.quiet and k == 0))
        allres.append(summarize(runs, clients))
        if k == 0:
            first_runs = runs

    names = list(first_runs.keys())
    print(f"\nRESULTS (mean +/- std over {a.repeats} seeds; global clean test set)")
    print(f"  {'arm':<15} {'accuracy':>15} {'worst vehicle':>14} {'rounds to 90%':>14} "
          f"{'uploaded KB':>12} {'dropped':>8}")
    for n in names:
        g = lambda key: np.array([r[n][key] for r in allres])
        print(f"  {n:<15} {g('acc').mean():>8.3f} +/-{g('acc').std():.3f} "
              f"{g('worst_veh').mean():>14.3f} {g('r90').mean():>14.1f} "
              f"{g('mb').mean():>12.2f} {g('dropped').mean():>8.1f}")
    print("\nHow to read this:")
    print("  * centralized is the ceiling; it needs all raw data on one server.")
    print("  * fedavg -> +selection / +quality isolates what each ingredient adds; differences")
    print("    can be small next to the std across seeds, so use --repeats 10+ before concluding.")
    print("  * proposed+topk: the toy model has only dim*classes weights, so index overhead")
    print("    limits the saving here; a model with millions of weights amortises it.")
    print("  * proposed+dp shows the accuracy price of the privacy noise.")
    print("  * 'worst vehicle' = accuracy of the global model on the hardest vehicle's data.")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "rounds.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "round", "test_loss", "test_acc", "participants", "dropped", "uploaded_kb"])
        for n, (_, h) in first_runs.items():
            for row in h:
                w.writerow([n, *row[:3], row[3], row[4], f"{row[5]:.6f}"])
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "acc_mean", "acc_std", "worst_vehicle_acc", "rounds_to_90pct",
                    "uploaded_kb", "dropped_uploads"])
        for n in names:
            g = lambda key: np.array([r[n][key] for r in allres])
            w.writerow([n, f"{g('acc').mean():.4f}", f"{g('acc').std():.4f}",
                        f"{g('worst_veh').mean():.4f}", f"{g('r90').mean():.1f}",
                        f"{g('mb').mean():.3f}", f"{g('dropped').mean():.1f}"])
    print(f"\nfiles: {path} (per-round, first seed), {os.path.join(a.out, 'summary.csv')}")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot (pip install matplotlib)")
            return
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for n, (_, h) in first_runs.items():
            ax[0].plot([x[0] for x in h], [x[2] for x in h], label=n)
            ax[1].plot([x[0] for x in h], np.cumsum([x[5] for x in h]), label=n)
        ax[0].set(xlabel="round", ylabel="global test accuracy", title="Convergence")
        ax[1].set(xlabel="round", ylabel="cumulative upload (KB)", title="Communication")
        ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "accuracy.png"), dpi=130)
        print("plot:", os.path.join(a.out, "accuracy.png"))


if __name__ == "__main__":
    main()
