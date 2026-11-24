"""OpenML vowel dataset (Deterding vowel recognition): loading, splits, OOD inputs.

Inputs are 12 features: speaker (ordinal code), sex (0/1), and 10 numeric
features. All 12 are standardized with the training split's mean and std.
Output is one of 11 vowel classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARFF_PATH = ROOT / "data" / "vowel.arff"

SPLIT_SEED = 0
OOD_SEED = 1
OOD_N = 1000
OOD_STD = 2.0  # OOD inputs: isotropic Gaussian with std 2 in standardized space


@dataclass
class Data:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_ood: np.ndarray
    feature_names: list[str]
    class_names: list[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def _parse_arff(path: Path):
    attrs: list[tuple[str, list[str] | None]] = []
    rows: list[list[str]] = []
    in_data = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if low.startswith("@attribute"):
            _, name, rest = line.split(None, 2)
            if rest.startswith("{"):
                values = [v.strip() for v in rest.strip("{}").split(",")]
                attrs.append((name, values))
            else:
                attrs.append((name, None))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append([v.strip() for v in line.split(",")])
    return attrs, rows


@lru_cache(maxsize=1)
def load_data() -> Data:
    attrs, rows = _parse_arff(ARFF_PATH)
    *feat_attrs, (class_attr, class_values) = attrs
    assert class_values is not None

    X = np.zeros((len(rows), len(feat_attrs)))
    for j, (_, values) in enumerate(feat_attrs):
        for i, row in enumerate(rows):
            X[i, j] = values.index(row[j]) if values is not None else float(row[j])
    y = np.array([class_values.index(row[-1]) for row in rows])

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(rows))
    n_train, n_val = int(0.6 * len(rows)), int(0.2 * len(rows))
    tr, va, te = perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]

    mean, std = X[tr].mean(0), X[tr].std(0)
    Xs = (X - mean) / std

    ood = np.random.default_rng(OOD_SEED).normal(0.0, OOD_STD, size=(OOD_N, X.shape[1]))

    return Data(
        X_train=Xs[tr], y_train=y[tr],
        X_val=Xs[va], y_val=y[va],
        X_test=Xs[te], y_test=y[te],
        X_ood=ood,
        feature_names=[name for name, _ in feat_attrs],
        class_names=list(class_values),
    )
