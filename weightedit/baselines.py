"""Baselines: structured magnitude pruning and standard distillation."""

from __future__ import annotations

import numpy as np

from .data import Data
from .mlp import init_mlp, layer_sizes, n_layers, softmax, forward, train
from .metrics import evaluate

DISTILL_STEPS = 3000
DISTILL_EVAL_STEPS = [0, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
DISTILL_SEEDS = [0, 1, 2]
DISTILL_LR = 5e-3
DISTILL_BATCH = 64


def student_sizes(teacher: dict) -> list[int]:
    s = layer_sizes(teacher)
    return [s[0]] + [h // 2 for h in s[1:-1]] + [s[-1]]


def magnitude_prune(teacher: dict) -> dict:
    """Keep the half of each hidden layer's units with the largest ||incoming|| * ||outgoing||.

    Layers are pruned in order; a unit's incoming norm only counts inputs that were kept.
    """
    L = n_layers(teacher)
    W = {i: teacher[f"W{i}"].copy() for i in range(1, L + 1)}
    b = {i: teacher[f"b{i}"].copy() for i in range(1, L + 1)}
    for i in range(1, L):
        h = W[i].shape[1]
        score = np.linalg.norm(W[i], axis=0) * np.linalg.norm(W[i + 1], axis=1)
        keep = np.sort(np.argsort(-score)[: h // 2])
        W[i], b[i] = W[i][:, keep], b[i][keep]
        W[i + 1] = W[i + 1][keep, :]
    out = {}
    for i in range(1, L + 1):
        out[f"W{i}"], out[f"b{i}"] = W[i], b[i]
    return out


def distillation_curve(teacher: dict, data: Data) -> dict:
    """Distill a fresh half-width student on the teacher's soft labels (training inputs only).

    Returns {step: {metric_mean, metric_std, ...}} averaged over DISTILL_SEEDS.
    """
    sizes = student_sizes(teacher)
    T = softmax(forward(teacher, data.X_train))
    per_seed = []
    for seed in DISTILL_SEEDS:
        rng = np.random.default_rng(1000 + seed)
        init = init_mlp(sizes, rng)
        _, evals = train(
            init, data.X_train, T, DISTILL_STEPS, rng,
            lr=DISTILL_LR, batch_size=DISTILL_BATCH,
            eval_steps=DISTILL_EVAL_STEPS,
            eval_fn=lambda p: evaluate(p, teacher, data),
        )
        per_seed.append(evals)
    keys = ["test_agreement", "test_kl", "ood_agreement", "ood_kl", "test_accuracy", "error_agreement"]
    curve = {}
    for step in DISTILL_EVAL_STEPS:
        row = {}
        for k in keys:
            vals = [e[step][k] for e in per_seed if e[step].get(k) is not None]
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{k}_std"] = float(np.std(vals)) if vals else None
        curve[str(step)] = row
    return curve
