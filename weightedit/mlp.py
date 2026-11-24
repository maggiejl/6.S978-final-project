"""A minimal numpy MLP: h_l = relu(h_{l-1} @ W_l + b_l), logits = h @ W_L + b_L.

Parameters are a dict {"W1": (in, h1), "b1": (h1,), ..., "WL": (h, out), "bL": (out,)}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def init_mlp(sizes: list[int], rng: np.random.Generator) -> dict[str, np.ndarray]:
    params = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
        params[f"W{i}"] = rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b))
        params[f"b{i}"] = np.zeros(b)
    return params


def n_layers(params: dict) -> int:
    return sum(1 for k in params if k.startswith("W"))


def layer_sizes(params: dict) -> list[int]:
    L = n_layers(params)
    return [params["W1"].shape[0]] + [params[f"W{i}"].shape[1] for i in range(1, L + 1)]


def n_params(params: dict) -> int:
    return int(sum(v.size for v in params.values()))


def param_shapes(sizes: list[int]) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
        shapes[f"W{i}"] = (a, b)
        shapes[f"b{i}"] = (b,)
    return shapes


def forward(params: dict, X: np.ndarray, return_cache: bool = False):
    L = n_layers(params)
    h = X
    cache = [X]
    for i in range(1, L + 1):
        z = h @ params[f"W{i}"] + params[f"b{i}"]
        h = np.maximum(z, 0.0) if i < L else z
        cache.append(h)
    return (h, cache) if return_cache else h


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def predict(params: dict, X: np.ndarray) -> np.ndarray:
    return forward(params, X).argmax(axis=1)


def _grads(params: dict, X: np.ndarray, T: np.ndarray, weight_decay: float):
    """Gradients of mean cross-entropy against soft targets T (= KL to T up to a constant)."""
    L = n_layers(params)
    logits, cache = forward(params, X, return_cache=True)
    d = (softmax(logits) - T) / len(X)
    grads = {}
    for i in range(L, 0, -1):
        h_prev = cache[i - 1]
        grads[f"W{i}"] = h_prev.T @ d + weight_decay * params[f"W{i}"]
        grads[f"b{i}"] = d.sum(axis=0)
        if i > 1:
            d = (d @ params[f"W{i}"].T) * (h_prev > 0)
    return grads


def train(
    params: dict,
    X: np.ndarray,
    T: np.ndarray,
    steps: int,
    rng: np.random.Generator,
    lr: float = 5e-3,
    batch_size: int = 64,
    weight_decay: float = 0.0,
    eval_steps: list[int] | None = None,
    eval_fn=None,
):
    """Adam on minibatches. T are target probabilities (one-hot labels or teacher softmax).

    If eval_steps/eval_fn are given, returns {step: eval_fn(params)} (step 0 = before training).
    """
    params = {k: v.copy() for k, v in params.items()}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    evals = {}
    eval_set = set(eval_steps or [])
    if 0 in eval_set and eval_fn:
        evals[0] = eval_fn(params)
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(X), size=batch_size)
        g = _grads(params, X[idx], T[idx], weight_decay)
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh = m[k] / (1 - b1**step)
            vh = v[k] / (1 - b2**step)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)
        if step in eval_set and eval_fn:
            evals[step] = eval_fn(params)
    return params, evals


# ---------- text formatting (what the LLM sees) ----------

def fmt_num(x: float, sig: int = 3) -> str:
    s = f"{x:.{sig}g}"
    return "0" if s in ("-0", "0") else s


def round_sig(params: dict, sig: int = 3) -> dict:
    return {k: np.vectorize(lambda x: float(fmt_num(x, sig)))(v).astype(float) for k, v in params.items()}


def fmt_array(a: np.ndarray, sig: int = 3) -> str:
    if a.ndim == 1:
        return "[" + ", ".join(fmt_num(x, sig) for x in a) + "]"
    return "[\n" + ",\n".join("  " + fmt_array(row, sig) for row in a) + "\n]"


# ---------- io ----------

def save_params(params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.tolist() for k, v in params.items()}))


def load_params(path: Path) -> dict:
    return {k: np.array(v, dtype=float) for k, v in json.loads(path.read_text()).items()}
