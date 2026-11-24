"""Train all teachers and compute baselines. Usage: python -m weightedit.prepare [--archs h8,h16]"""

from __future__ import annotations

import argparse
import json

from .baselines import DISTILL_STEPS, distillation_curve, magnitude_prune, student_sizes
from .data import load_data
from .metrics import evaluate, unit_origins
from .mlp import layer_sizes, n_params, predict, save_params
from .teachers import ARCHS, SIG_FIGS, TEACHER_DIR, rounding_check, train_teacher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default=",".join(ARCHS))
    args = ap.parse_args()

    data = load_data()
    print(f"data: train={len(data.X_train)} val={len(data.X_val)} test={len(data.X_test)} ood={len(data.X_ood)}")
    for arch in args.archs.split(","):
        teacher = train_teacher(ARCHS[arch], data)
        d = TEACHER_DIR / arch
        save_params(teacher, d / "teacher.json")

        rc = rounding_check(teacher, data)
        meta = {
            "arch": arch,
            "sizes": layer_sizes(teacher),
            "n_params": n_params(teacher),
            "student_sizes": student_sizes(teacher),
            "train_accuracy": float((predict(teacher, data.X_train) == data.y_train).mean()),
            "test_accuracy": float((predict(teacher, data.X_test) == data.y_test).mean()),
            "sig_figs": SIG_FIGS,
            "rounding_check": rc,
        }
        pruned = magnitude_prune(teacher)
        pruned_metrics = evaluate(pruned, teacher, data)
        pruned_metrics["n_params"] = n_params(pruned)
        pruned_metrics["unit_origins"] = unit_origins(pruned, teacher)
        curve = distillation_curve(teacher, data)
        baselines = {
            "magnitude_pruning": pruned_metrics,
            "distillation_curve": curve,
            "distillation_ceiling": curve[str(DISTILL_STEPS)],
        }
        meta["student_n_params"] = n_params(pruned)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        (d / "baselines.json").write_text(json.dumps(baselines, indent=2))

        ceil = curve[str(DISTILL_STEPS)]["test_agreement_mean"]
        print(
            f"{arch:7s} params={meta['n_params']:5d} student={meta['student_n_params']:5d} "
            f"teacher_acc={meta['test_accuracy']:.3f} rounding_changed={sum(rc.values())} "
            f"prune_agree={pruned_metrics['test_agreement']:.3f} distill@{DISTILL_STEPS}={ceil:.3f}"
        )


if __name__ == "__main__":
    main()
