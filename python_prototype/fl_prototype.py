"""Minimal federated-learning prototype for collaborative AV training (numpy only).

A fleet of vehicles trains a shared softmax classifier (stand-in for a perception
model) on private, non-IID, quality-varying data. Only weight deltas reach the RSU.

Compares three arms:  centralized (upper bound) | FedAvg | proposed
(connectivity-aware selection + quality-aware aggregation).

Run:  python python_prototype/fl_prototype.py [--clients 20 --rounds 30 --seed 1]
"""
import argparse
import csv
import os

import numpy as np


# ----------------------------------------------------------------- data ----
def make_fleet(rng, n_clients, n_classes, dim, alpha, samples):
    """Synthetic sensor features. Non-IID: Dirichlet label skew, log-normal
    quantity skew, per-vehicle feature shift (region) and noise (sensor quality)."""
    means = rng.normal(0, 0.6, (n_classes, dim))
    clients = []
    for _ in range(n_clients):
        n = max(20, int(samples * rng.lognormal(0, 0.4)))
        p = rng.dirichlet(alpha * np.ones(n_classes))
        y = rng.choice(n_classes, n, p=p)
        noise = rng.uniform(0.8, 2.5)                    # sensor degradation
        shift = rng.normal(0, 0.5, dim)                  # region / weather shift
        X = means[y] + shift + rng.normal(0, noise, (n, dim))
        clients.append(dict(X=X, y=y, noise=noise, uptime=rng.uniform(0.2, 1.0)))
    yt = rng.integers(0, n_classes, 1000)
    Xt = means[yt] + rng.normal(0, 1.0, (1000, dim))     # clean global test set
    return clients, (Xt, yt)


# ---------------------------------------------------------------- model ----
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def loss_acc(W, X, y):
    p = softmax(X @ W)
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean(), (p.argmax(1) == y).mean()


def local_train(W, X, y, rng, epochs=2, lr=0.05, batch=32):
    W = W.copy()
    for _ in range(epochs):
        idx = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            b = idx[s:s + batch]
            p = softmax(X[b] @ W)
            p[np.arange(len(b)), y[b]] -= 1
            W -= lr * X[b].T @ p / len(b)
    return W


# ---------------------------------------------------------- federation -----
def select(rng, clients, frac, uptime_min, aware):
    k = max(2, int(np.ceil(frac * len(clients))))
    if not aware:
        return list(rng.choice(len(clients), k, replace=False))
    up = np.array([c["uptime"] for c in clients])
    ok = np.flatnonzero(up >= uptime_min)
    if len(ok) < k:
        ok = np.argsort(-up)[:k]
    return list(rng.choice(ok, k, replace=False, p=up[ok] / up[ok].sum()))


def run_fl(clients, test, dim, n_classes, rounds, frac, aware, seed, epochs=2):
    rng = np.random.default_rng(seed)
    W = np.zeros((dim, n_classes))
    hist = []
    for r in range(1, rounds + 1):
        sel = select(rng, clients, frac, 0.55, aware)
        deltas, weights = [], []
        for i in sel:
            c = clients[i]
            if rng.random() > c["uptime"]:               # V2I dropout
                continue
            Wk = local_train(W, c["X"], c["y"], rng, epochs)
            n = len(c["y"])
            q = 1.0 / (1.0 + loss_acc(Wk, c["X"], c["y"])[0]) if aware else 1.0
            deltas.append(Wk - W)
            weights.append(q * n)
        if deltas:
            w = np.array(weights) / sum(weights)
            W = W + sum(wi * d for wi, d in zip(w, deltas))
        l, a = loss_acc(W, *test)
        hist.append((r, l, a, len(deltas)))
    return W, hist


def run_central(clients, test, dim, n_classes, rounds, seed, epochs=2):
    rng = np.random.default_rng(seed)
    X = np.vstack([c["X"] for c in clients])
    y = np.concatenate([c["y"] for c in clients])
    W = np.zeros((dim, n_classes))
    hist = []
    for r in range(1, rounds + 1):
        W = local_train(W, X, y, rng, epochs)
        l, a = loss_acc(W, *test)
        hist.append((r, l, a, len(clients)))
    return W, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.4, help="Dirichlet skew")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    clients, test = make_fleet(rng, a.clients, a.classes, a.dim, a.alpha, 150)
    arms = {
        "centralized": run_central(clients, test, a.dim, a.classes, a.rounds, a.seed)[1],
        "fedavg": run_fl(clients, test, a.dim, a.classes, a.rounds, a.frac, False, a.seed)[1],
        "proposed": run_fl(clients, test, a.dim, a.classes, a.rounds, a.frac, True, a.seed)[1],
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "rounds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "round", "test_loss", "test_acc", "participants"])
        for name, h in arms.items():
            for row in h:
                w.writerow([name, *row])
    print(f"{a.clients} vehicles, {a.rounds} rounds, Dirichlet alpha={a.alpha}")
    for name, h in arms.items():
        print(f"  {name:12s} final acc = {h[-1][2]:.3f}   final loss = {h[-1][1]:.3f}")
    print(f"per-round metrics -> {os.path.join(a.out, 'rounds.csv')}")


if __name__ == "__main__":
    main()
