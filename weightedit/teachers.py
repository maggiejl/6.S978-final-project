"""Teacher architectures, training, and the prepared artifacts (teacher weights + baselines).

Run `python -m weightedit.prepare` once to train every teacher and compute baselines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import Data, load_data
from .mlp import init_mlp, load_params, predict, round_sig, train

ROOT = Path(__file__).resolve().parent.parent
TEACHER_DIR = ROOT / "artifacts" / "teachers"

# name -> hidden layer widths
ARCHS: dict[str, list[int]] = {
    "h8": [8],
    "h16": [16],
    "h32": [32],
    "h64": [64],
    "h16-16": [16, 16],  # depth side experiment
}

TEACHER_SEED = 0
TEACHER_STEPS = 6000
TEACHER_LR = 5e-3
TEACHER_BATCH = 64
TEACHER_WD = 1e-4
SIG_FIGS = 3


def train_teacher(hidden: list[int], data: Data) -> dict:
    rng = np.random.default_rng(TEACHER_SEED)
    sizes = [data.n_features, *hidden, data.n_classes]
    onehot = np.eye(data.n_classes)[data.y_train]
    params, _ = train(
        init_mlp(sizes, rng), data.X_train, onehot, TEACHER_STEPS, rng,
        lr=TEACHER_LR, batch_size=TEACHER_BATCH, weight_decay=TEACHER_WD,
    )
    return params


def rounding_check(teacher: dict, data: Data) -> dict:
    """Does rounding the teacher to SIG_FIGS change any prediction?"""
    r = round_sig(teacher, SIG_FIGS)
    out = {}
    for name, X in [("train", data.X_train), ("val", data.X_val), ("test", data.X_test), ("ood", data.X_ood)]:
        out[f"{name}_changed"] = int((predict(teacher, X) != predict(r, X)).sum())
    return out


@dataclass
class Teacher:
    arch: str
    params: dict          # full-precision teacher; the reference for all metrics
    meta: dict
    baselines: dict

    @property
    def rounded(self) -> dict:
        return round_sig(self.params, SIG_FIGS)


@lru_cache(maxsize=None)
def load_teacher(arch: str) -> Teacher:
    d = TEACHER_DIR / arch
    if not (d / "teacher.json").exists():
        raise FileNotFoundError(f"No teacher for {arch}. Run `python -m weightedit.prepare` first.")
    return Teacher(
        arch=arch,
        params=load_params(d / "teacher.json"),
        meta=json.loads((d / "meta.json").read_text()),
        baselines=json.loads((d / "baselines.json").read_text()),
    )


def get_data() -> Data:
    return load_data()
