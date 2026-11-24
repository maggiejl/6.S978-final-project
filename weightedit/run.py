"""Run an experiment. Each episode is saved to results/runs/<run_id>.json and skipped if it already exists,
so experiments resume after interruption and share runs that have identical settings.

Examples:
  python -m weightedit.run exp0 --models gpt-4o            # pilot: the three conditions
  python -m weightedit.run exp1                            # model capability x teacher width
  python -m weightedit.run exp1_depth --models o3          # 1 vs 2 hidden layers, best model
  python -m weightedit.run exp2                            # model capability x harness
  python -m weightedit.run exp1 --list                     # show planned runs, don't call any API
  python -m weightedit.run exp1 --dry-run                  # whole pipeline with the offline mock model
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .harness import run_episode
from .llm import make_client
from .models import MODEL_ORDER, MODELS

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunSpec:
    model: str
    arch: str
    condition: str
    harness: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.model}__{self.arch}__{self.condition}__{self.harness}__s{self.seed}"


def plan(exp: str, models: list[str]) -> list[RunSpec]:
    if exp == "exp0":  # pilot: which inputs does the LLM need? one model, 16-unit teacher
        return [RunSpec(m, "h16", c, "one_shot", s)
                for m in models for c in ["task_only", "weights_only", "weights_and_task"] for s in range(5)]
    if exp == "exp1":  # model capability x teacher width, one shot
        return [RunSpec(m, a, "weights_only", "one_shot", s)
                for m in models for a in ["h8", "h16", "h32", "h64"] for s in range(5)]
    if exp == "exp1_depth":  # side experiment: 1 vs 2 hidden layers, best model only
        return [RunSpec(m, a, "weights_only", "one_shot", s)
                for m in models for a in ["h16", "h16-16"] for s in range(5)]
    if exp == "exp2":  # model capability x harness, 16-unit teacher
        return [RunSpec(m, "h16", "weights_only", h, s)
                for m in models for h in ["one_shot", "multi_turn"] for s in range(3)]
    raise ValueError(f"unknown experiment {exp}")


def load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=["exp0", "exp1", "exp1_depth", "exp2"])
    ap.add_argument("--models", help="comma-separated short names from weightedit/models.py (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="run at most this many new episodes")
    ap.add_argument("--list", action="store_true", help="print planned runs and exit")
    ap.add_argument("--dry-run", action="store_true", help="use the offline mock model; writes to results_dryrun/")
    ap.add_argument("--rerun-failed", action="store_true", help="redo saved runs that ended in an API error")
    args = ap.parse_args()

    load_dotenv()
    if args.dry_run:
        models = ["mock"]
        out_dir = ROOT / "results_dryrun" / "runs"
    else:
        models = args.models.split(",") if args.models else MODEL_ORDER
        out_dir = ROOT / "results" / "runs"
        if args.exp in ("exp0", "exp1_depth") and not args.models:
            ap.error(f"{args.exp} uses a single model: pass --models")
        bad = [m for m in models if m not in MODELS or MODELS[m]["id"] == "TODO"]
        if bad and not args.list:
            ap.error(f"set an OpenRouter id in weightedit/models.py for: {bad}")
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = plan(args.exp, models)
    todo = []
    for s in specs:
        path = out_dir / f"{s.run_id}.json"
        if path.exists():
            if not args.rerun_failed:
                continue
            rec = json.loads(path.read_text())
            if not any("api_error" in t for t in rec["turns"]):
                continue
        todo.append(s)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.exp}: {len(specs)} runs planned, {len(specs) - len(todo)} already saved, {len(todo)} to run")
    if args.list:
        for s in todo:
            print("  ", s.run_id)
        return

    def work(spec: RunSpec):
        rec = run_episode(make_client(spec.model), spec.model, spec.arch, spec.condition, spec.harness, spec.seed)
        (out_dir / f"{spec.run_id}.json").write_text(json.dumps(rec, indent=1))
        return rec

    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, s): s for s in todo}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                rec = fut.result()
            except Exception:
                print(f"[crash] {spec.run_id}\n{traceback.format_exc()}")
                continue
            total_cost += rec["cost_usd"]
            f = rec["final"]
            summary = (f"test_agree={f['test_agreement']:.3f} ({100 * f['pct_of_distill_ceiling']:.0f}% of distill) "
                       f"copied={f['unit_origins']['n_copied']}/{f['unit_origins']['n_units']}"
                       if f["valid"] else f"INVALID: {f['reason'][:120]}")
            print(f"[done] {spec.run_id}: {summary} turns={rec['n_turns']} cost=${rec['cost_usd']:.4f}")
    print(f"total cost this invocation: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
