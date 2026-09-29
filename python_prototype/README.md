# Python prototype

Self-contained, numpy-only federated-learning demo (no PyTorch). For the full
framework see the repository root (`fedav/`).

    pip install numpy
    python python_prototype/fl_prototype.py --clients 20 --rounds 30 --seed 1

Compares centralized, FedAvg and the proposed method (connectivity-aware client
selection + quality-aware aggregation) on synthetic non-IID vehicle data.
Writes `results/rounds.csv`.
