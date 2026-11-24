"""Aggregate saved runs into a CSV, summary tables, and figures.

Usage: python -m weightedit.analyze [--dry-run]
Writes <results>/summary.csv and <results>/figures/*.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .models import MODEL_ORDER
from .teachers import ARCHS, load_teacher

ROOT = Path(__file__).resolve().parent.parent

FIELDS = ["model", "arch", "condition", "harness", "seed", "valid", "test_agreement", "test_kl",
          "ood_agreement", "ood_kl", "test_accuracy", "error_agreement", "pct_of_distill_ceiling",
          "beats_pruning", "equivalent_distill_steps", "n_copied", "n_new", "n_turns", "final_turn",
          "prompt_tokens", "completion_tokens", "cost_usd", "failure"]


def load_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(runs_dir.glob("*.json")):
        r = json.loads(p.read_text())
        f = r["final"]
        row = {k: r.get(k) for k in ["model", "arch", "condition", "harness", "seed", "n_turns", "final_turn",
                                     "prompt_tokens", "completion_tokens", "cost_usd"]}
        row["valid"] = f["valid"]
        row["failure"] = None if f["valid"] else f.get("reason")
        for k in ["test_agreement", "test_kl", "ood_agreement", "ood_kl", "test_accuracy", "error_agreement",
                  "pct_of_distill_ceiling", "beats_pruning", "equivalent_distill_steps"]:
            row[k] = f.get(k)
        row["n_copied"] = f.get("unit_origins", {}).get("n_copied")
        row["n_new"] = f.get("unit_origins", {}).get("n_new")
        rows.append(row)
    return rows


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


def table(rows: list[dict], keys: tuple[str, ...], title: str) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    if not groups:
        return
    print(f"\n## {title}")
    print(" | ".join([*keys, "n", "valid", "test_agree", "% distill ceiling", "ood_agree", "err_agree", "copied units"]))
    def order(g):
        return tuple((MODEL_ORDER.index(x), "") if x in MODEL_ORDER else (len(MODEL_ORDER), str(x)) for x in g)

    for g in sorted(groups, key=order):
        rs = groups[g]
        v = [r for r in rs if r["valid"]]
        print(" | ".join([*map(str, g), str(len(rs)), f"{len(v)}/{len(rs)}",
                          _mean_std([r["test_agreement"] for r in v]),
                          _mean_std([r["pct_of_distill_ceiling"] for r in v]),
                          _mean_std([r["ood_agreement"] for r in v]),
                          _mean_std([r["error_agreement"] for r in v]),
                          _mean_std([r["n_copied"] for r in v])]))


def baselines_table() -> None:
    print("\n## Baselines (test agreement with teacher)")
    print("arch | teacher params | student params | teacher acc | magnitude pruning | distill@3000")
    for arch in ARCHS:
        try:
            t = load_teacher(arch)
        except FileNotFoundError:
            continue
        b = t.baselines
        print(f"{arch} | {t.meta['n_params']} | {t.meta['student_n_params']} | {t.meta['test_accuracy']:.3f} | "
              f"{b['magnitude_pruning']['test_agreement']:.3f} | {b['distillation_ceiling']['test_agreement_mean']:.3f}")


def plot_exp1(rows: list[dict], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = {"h8": 8, "h16": 16, "h32": 32, "h64": 64}
    rs = [r for r in rows if r["harness"] == "one_shot" and r["condition"] == "weights_only" and r["arch"] in widths]
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rs)] or sorted({r["model"] for r in rs})
    for m in models:
        xs, ys, es = [], [], []
        for a, w in widths.items():
            v = [r["pct_of_distill_ceiling"] for r in rs if r["model"] == m and r["arch"] == a and r["valid"]]
            if v:
                xs.append(w); ys.append(100 * np.mean(v)); es.append(100 * np.std(v))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=m)
    px, py = [], []
    for a, w in widths.items():
        try:
            b = load_teacher(a).baselines
        except FileNotFoundError:
            continue
        px.append(w)
        py.append(100 * b["magnitude_pruning"]["test_agreement"] / b["distillation_ceiling"]["test_agreement_mean"])
    ax.plot(px, py, "k--", label="magnitude pruning")
    ax.axhline(100, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(widths.values()), [str(w) for w in widths.values()])
    ax.set_xlabel("teacher hidden width")
    ax.set_ylabel("test agreement (% of distillation ceiling)")
    ax.set_title("Exp 1: one-shot compression vs teacher width")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "exp1_width.png", dpi=200)


def plot_exp2(rows: list[dict], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [r for r in rows if r["arch"] == "h16" and r["condition"] == "weights_only" and r["seed"] < 3]
    if not any(r["harness"] == "multi_turn" for r in rs):
        return
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rs)] or sorted({r["model"] for r in rs})
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models))
    for i, h in enumerate(["one_shot", "multi_turn"]):
        ys, es = [], []
        for m in models:
            v = [r["pct_of_distill_ceiling"] for r in rs if r["model"] == m and r["harness"] == h and r["valid"]]
            ys.append(100 * np.mean(v) if v else 0); es.append(100 * np.std(v) if v else 0)
        ax.bar(x + (i - 0.5) * 0.38, ys, 0.38, yerr=es, capsize=3, label=h)
    ax.set_xticks(x, models)
    ax.axhline(100, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("test agreement (% of distillation ceiling)")
    ax.set_title("Exp 2: harness vs model (16-unit teacher)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_harness.png", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    res = ROOT / ("results_dryrun" if args.dry_run else "results")
    rows = load_rows(res / "runs")
    print(f"{len(rows)} runs")

    res.mkdir(exist_ok=True)
    with open(res / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    baselines_table()
    table([r for r in rows if r["arch"] == "h16" and r["harness"] == "one_shot"], ("model", "condition"),
          "Exp 0: conditions (h16, one shot)")
    table([r for r in rows if r["condition"] == "weights_only" and r["harness"] == "one_shot"], ("model", "arch"),
          "Exp 1: model x teacher (one shot)")
    table([r for r in rows if r["condition"] == "weights_only" and r["arch"] == "h16" and r["seed"] < 3],
          ("model", "harness"), "Exp 2: model x harness (h16, seeds 0-2)")

    fails = [r for r in rows if not r["valid"]]
    if fails:
        print("\n## Invalid runs")
        for r in fails:
            print(f"{r['model']} {r['arch']} {r['condition']} {r['harness']} s{r['seed']}: {str(r['failure'])[:150]}")

    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_exp1(rows, fig_dir)
    plot_exp2(rows, fig_dir)
    print(f"\nwrote {res / 'summary.csv'} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
