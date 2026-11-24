"""Harnesses: one-shot (no tools) and multi-turn with validation-agreement feedback."""

from __future__ import annotations

import time

from .metrics import equivalent_steps, evaluate, unit_origins, val_agreement
from .mlp import n_params
from .models import MODELS
from .prompts import SYSTEM, ParseError, feedback_prompt, initial_prompt, parse_student
from .teachers import load_teacher, get_data

HARNESSES = {"one_shot": 1, "multi_turn": 5}  # name -> max attempts


def score_student(student: dict, teacher) -> dict:
    data = get_data()
    m = evaluate(student, teacher.params, data)
    if not m["valid"]:
        return m
    base = teacher.baselines
    ceiling = base["distillation_ceiling"]["test_agreement_mean"]
    m["pct_of_distill_ceiling"] = m["test_agreement"] / ceiling
    m["prune_test_agreement"] = base["magnitude_pruning"]["test_agreement"]
    m["beats_pruning"] = m["test_agreement"] > base["magnitude_pruning"]["test_agreement"]
    m["equivalent_distill_steps"] = equivalent_steps(m["test_agreement"], base["distillation_curve"])
    m["val_agreement"] = val_agreement(student, teacher.params, data)
    m["unit_origins"] = unit_origins(student, teacher.params)
    m["n_params"] = n_params(student)
    return m


def run_episode(client, model: str, arch: str, condition: str, harness: str, seed: int) -> dict:
    """Run one LLM episode and return a JSON-serializable record (prompts, responses, metrics)."""
    cfg = MODELS[model]
    teacher = load_teacher(arch)
    data = get_data()
    s_sizes = teacher.meta["student_sizes"]
    max_turns = HARNESSES[harness]

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": initial_prompt(teacher.rounded, s_sizes, condition, data,
                                                   multi_turn=max_turns > 1, max_turns=max_turns)},
    ]
    turns = []
    best = None  # (val_agreement, turn index)
    for attempt in range(1, max_turns + 1):
        t0 = time.time()
        turn = {"attempt": attempt}
        try:
            resp = client.chat(cfg["id"], messages, cfg["max_tokens"], seed=seed,
                               context={"teacher": teacher.params, "student_sizes": s_sizes})
        except Exception as e:  # API error (e.g. context length): episode ends here
            turn.update({"api_error": str(e), "seconds": time.time() - t0})
            turns.append(turn)
            break
        turn.update({k: resp[k] for k in ("content", "reasoning", "finish_reason", "usage", "provider", "model")})
        turn["seconds"] = time.time() - t0
        messages.append({"role": "assistant", "content": resp["content"]})

        error = None
        try:
            student = parse_student(resp["content"], s_sizes)
            metrics = score_student(student, teacher)
            if not metrics["valid"]:
                error = metrics["reason"]
            else:
                turn["student"] = {k: v.tolist() for k, v in student.items()}
        except ParseError as e:
            error, metrics = str(e), {"valid": False, "reason": str(e)}
        turn["metrics"] = metrics
        turns.append(turn)

        if metrics["valid"] and (best is None or metrics["val_agreement"] > best[0]):
            best = (metrics["val_agreement"], len(turns) - 1)
        if attempt == max_turns or (metrics["valid"] and metrics["val_agreement"] >= 1.0):
            break
        messages.append({"role": "user", "content": feedback_prompt(
            attempt, max_turns, metrics.get("val_agreement"), len(data.X_val), error)})

    last_valid = next((i for i in range(len(turns) - 1, -1, -1) if turns[i].get("metrics", {}).get("valid")), None)
    usage = [t.get("usage") or {} for t in turns]
    return {
        "model": model,
        "model_id": cfg["id"],
        "arch": arch,
        "condition": condition,
        "harness": harness,
        "seed": seed,
        "student_sizes": s_sizes,
        "messages": messages,
        "turns": turns,
        "n_turns": len(turns),
        # Primary student: best validation agreement across attempts (= the only attempt for one-shot).
        "final": turns[best[1]]["metrics"] if best else {"valid": False, "reason": _failure_reason(turns)},
        "final_turn": best[1] + 1 if best else None,
        "last_valid": turns[last_valid]["metrics"] if last_valid is not None else None,
        "prompt_tokens": sum(u.get("prompt_tokens", 0) or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens", 0) or 0 for u in usage),
        "cost_usd": sum(u.get("cost", 0) or 0 for u in usage),
    }


def _failure_reason(turns: list[dict]) -> str:
    t = turns[-1]
    if "api_error" in t:
        return "api_error: " + t["api_error"][:300]
    return t.get("metrics", {}).get("reason", "unknown")
