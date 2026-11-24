"""Prompts for the three conditions and the multi-turn feedback, plus answer parsing."""

from __future__ import annotations

import json
import re

import numpy as np

from .data import Data
from .mlp import fmt_array, layer_sizes, n_layers, param_shapes

CONDITIONS = ["task_only", "weights_only", "weights_and_task"]

SYSTEM = "You are an expert in neural networks."

TASK_DESCRIPTION = """\
Task: the network classifies spoken vowels (the Deterding vowel recognition data). Each example is
one vowel sound spoken by one speaker. The 12 input features are, in order:
  0: speaker identity (integer code 0-14, standardized)
  1: speaker sex (0 = male, 1 = female, standardized)
  2-11: ten acoustic features computed from the recording (log-area ratios from LPC analysis)
The 11 output classes are the vowels in the words, in order:
  {classes}
(each class name is h + vowel + d, e.g. "hid" = the vowel in "heed")."""


def _arch_str(sizes: list[int]) -> str:
    return " -> ".join(str(s) for s in sizes)


def _shapes_str(sizes: list[int]) -> str:
    return ", ".join(f'{k} {list(v)}' for k, v in param_shapes(sizes).items())


def _weights_block(teacher_rounded: dict) -> str:
    parts = []
    for i in range(1, n_layers(teacher_rounded) + 1):
        W, b = teacher_rounded[f"W{i}"], teacher_rounded[f"b{i}"]
        parts.append(f"W{i} (shape {W.shape[0]} x {W.shape[1]}; row = input unit, column = output unit):\n{fmt_array(W)}")
        parts.append(f"b{i} (length {b.shape[0]}):\n{fmt_array(b)}")
    return "\n\n".join(parts)


def initial_prompt(teacher_rounded: dict, student_sizes: list[int], condition: str, data: Data,
                   multi_turn: bool, max_turns: int) -> str:
    assert condition in CONDITIONS
    t_sizes = layer_sizes(teacher_rounded)
    L = len(t_sizes) - 1
    forward_str = "; ".join(
        [f"h{i} = relu(h{i-1} @ W{i} + b{i})" for i in range(1, L)] + [f"logits = h{L-1} @ W{L} + b{L}"]
    ).replace("h0", "x")

    lines = [
        "A small neural network classifier (the \"teacher\") has already been trained. Your job is to write "
        "the weights of a smaller \"student\" network that makes the same predictions as the teacher on any input.",
        "",
        f"Both networks are MLPs with ReLU hidden layers: {forward_str}; prediction = argmax(logits). "
        "x is a row vector of input features. Inputs are standardized (mean 0, std 1 on the training data).",
        "",
        f"Teacher architecture: {_arch_str(t_sizes)}",
        f"Student architecture: {_arch_str(student_sizes)} (half the teacher's hidden units)",
    ]
    if condition in ("task_only", "weights_and_task"):
        lines += ["", TASK_DESCRIPTION.format(classes=", ".join(data.class_names))]
    if condition in ("weights_only", "weights_and_task"):
        lines += ["", "Teacher weights (rounded to 3 significant figures):", "", _weights_block(teacher_rounded)]
    else:
        lines += ["", "You will not see the teacher's weights."]

    lines += [
        "",
        "Rules: you cannot run code, you have no training data, and you cannot query the teacher.",
    ]
    if multi_turn:
        lines.append(
            f"You get up to {max_turns} attempts. After each attempt you will be told how often your student "
            "agrees with the teacher's predictions on a held-out validation set, and you may revise it."
        )
    lines += [
        "",
        "Answer format: you may reason first, then give the complete student weights as one JSON object in a "
        "```json code block, with keys and shapes: " + _shapes_str(student_sizes) + ". "
        "Weight matrices are nested lists of rows (row = input unit). Only the last ```json block in your answer is used.",
    ]
    return "\n".join(lines)


def feedback_prompt(attempt: int, max_turns: int, val_agreement: float | None, n_val: int, error: str | None) -> str:
    remaining = max_turns - attempt
    if error is not None:
        head = f"Attempt {attempt} of {max_turns} could not be evaluated: {error}"
    else:
        head = (f"Attempt {attempt} of {max_turns}: your student agrees with the teacher's predictions on "
                f"{100 * val_agreement:.1f}% of {n_val} validation inputs.")
    return (f"{head}\nYou have {remaining} attempt(s) left. Revise the student and give the complete weights "
            "again in the same JSON format.")


# ---------- parsing ----------

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _last_json_object(text: str) -> str | None:
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1]
    # fallback: last balanced {...} that parses
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    cand = text[start : end + 1]
                    try:
                        json.loads(cand)
                        return cand
                    except json.JSONDecodeError:
                        break
        end = text.rfind("}", 0, end)
    return None


class ParseError(Exception):
    pass


def parse_student(text: str, student_sizes: list[int]) -> dict:
    raw = _last_json_object(text)
    if raw is None:
        raise ParseError("no JSON object found in the answer")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON ({e})")
    shapes = param_shapes(student_sizes)
    missing = [k for k in shapes if k not in obj]
    if missing:
        raise ParseError(f"missing keys {missing}")
    out = {}
    for k, shape in shapes.items():
        try:
            arr = np.array(obj[k], dtype=float)
        except (ValueError, TypeError):
            raise ParseError(f"{k} is not a numeric array of shape {list(shape)}")
        if arr.shape != shape:
            raise ParseError(f"{k} has shape {list(arr.shape)}, expected {list(shape)}")
        if not np.all(np.isfinite(arr)):
            raise ParseError(f"{k} contains non-finite values")
        out[k] = arr
    return out
