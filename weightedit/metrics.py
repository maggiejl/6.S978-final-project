"""Metrics comparing a student to its teacher."""

from __future__ import annotations

import numpy as np

from .data import Data
from .mlp import forward, log_softmax


def _kl(t_logits: np.ndarray, s_logits: np.ndarray) -> float:
    """Mean KL(teacher || student) over inputs."""
    lt, ls = log_softmax(t_logits), log_softmax(s_logits)
    return float((np.exp(lt) * (lt - ls)).sum(axis=1).mean())


def fidelity(student: dict, teacher: dict, X: np.ndarray) -> dict:
    t, s = forward(teacher, X), forward(student, X)
    return {
        "agreement": float((t.argmax(1) == s.argmax(1)).mean()),
        "kl": _kl(t, s),
    }


def evaluate(student: dict, teacher: dict, data: Data) -> dict:
    """All per-student metrics. Returns {"valid": False, ...} if outputs are non-finite."""
    for split_X in (data.X_test, data.X_ood):
        if not np.all(np.isfinite(forward(student, split_X))):
            return {"valid": False, "reason": "non-finite student outputs"}

    test = fidelity(student, teacher, data.X_test)
    ood = fidelity(student, teacher, data.X_ood)
    s_pred = forward(student, data.X_test).argmax(1)
    t_pred = forward(teacher, data.X_test).argmax(1)
    teacher_wrong = t_pred != data.y_test
    error_agreement = (
        float((s_pred[teacher_wrong] == t_pred[teacher_wrong]).mean()) if teacher_wrong.any() else None
    )
    return {
        "valid": True,
        "test_agreement": test["agreement"],
        "test_kl": test["kl"],
        "ood_agreement": ood["agreement"],
        "ood_kl": ood["kl"],
        "test_accuracy": float((s_pred == data.y_test).mean()),
        "error_agreement": error_agreement,
        "n_teacher_errors": int(teacher_wrong.sum()),
    }


def val_agreement(student: dict, teacher: dict, data: Data) -> float:
    """Feedback signal for the multi-turn harness (validation split only)."""
    return fidelity(student, teacher, data.X_val)["agreement"]


def equivalent_steps(agreement: float, curve: dict) -> float | None:
    """Fewest distillation steps whose mean test agreement reaches `agreement`.

    Linear interpolation between evaluated steps. Returns None if the curve never
    reaches it (i.e. the student beats distillation at the maximum step count).
    """
    steps = sorted(int(s) for s in curve)
    vals = [curve[str(s)]["test_agreement_mean"] if str(s) in curve else curve[s]["test_agreement_mean"] for s in steps]
    if agreement <= vals[0]:
        return 0.0
    for (s0, v0), (s1, v1) in zip(zip(steps, vals), zip(steps[1:], vals[1:])):
        if v1 >= agreement:
            return float(s0 + (s1 - s0) * (agreement - v0) / max(v1 - v0, 1e-12))
    return None


def unit_origins(student: dict, teacher: dict, cos_thresh: float = 0.99, norm_tol: float = 0.05) -> dict:
    """Classify each first-layer student hidden unit as copied from a teacher unit or new.

    A unit counts as copied if its incoming weights + bias have cosine similarity
    >= cos_thresh with some teacher unit and norms within norm_tol (relative).
    """
    S = np.vstack([student["W1"], student["b1"][None, :]]).T  # (h_s, in+1)
    T = np.vstack([teacher["W1"], teacher["b1"][None, :]]).T  # (h_t, in+1)
    Sn = np.linalg.norm(S, axis=1, keepdims=True)
    Tn = np.linalg.norm(T, axis=1, keepdims=True)
    cos = (S / np.maximum(Sn, 1e-12)) @ (T / np.maximum(Tn, 1e-12)).T
    best = cos.argmax(1)
    copied = []
    for j, k in enumerate(best):
        rel = abs(Sn[j, 0] - Tn[k, 0]) / max(Tn[k, 0], 1e-12)
        copied.append(bool(cos[j, k] >= cos_thresh and rel <= norm_tol))
    n_copied = int(sum(copied))
    return {
        "n_units": int(len(S)),
        "n_copied": n_copied,
        "n_new": int(len(S) - n_copied),
        "copied_from": [int(best[j]) if c else None for j, c in enumerate(copied)],
        "max_cos": [float(cos[j, best[j]]) for j in range(len(S))],
    }
