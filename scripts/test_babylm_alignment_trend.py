#!/usr/bin/env python
"""Test hypothesis: brain alignment decreases as BabyLM token exposure increases.

Input is the JSON produced by run_language_models.py (language_model_results.json).
The script performs:
1) Per-session slope of Fisher-z-transformed RSA correlation vs log(tokens)
2) A pooled exact permutation test across sessions (within-session label shuffles)

Usage example:
python scripts/test_babylm_alignment_trend.py \
  --results-json /path/to/language_model_results.json \
  --token-map '{"BrainAlign/gpt2-babylm-5":1.2e8,"BrainAlign/gpt2-babylm-7":1.8e8,"BrainAlign/gpt2-babylm-9":2.4e8}'
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


MODEL_STAGE_RE = re.compile(r"babylm-(\d+)")


def fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def parse_token_map(token_map_arg: str | None, models: List[str]) -> Dict[str, float]:
    if token_map_arg:
        parsed = json.loads(token_map_arg)
        return {k: float(v) for k, v in parsed.items()}

    token_map: Dict[str, float] = {}
    for m in models:
        match = MODEL_STAGE_RE.search(m)
        if not match:
            raise ValueError(
                f"Could not infer stage from model name '{m}'. "
                "Pass --token-map explicitly."
            )
        # Fallback proxy if exact token counts are not provided.
        token_map[m] = float(match.group(1))
    return token_map


def slope(x: np.ndarray, y: np.ndarray) -> float:
    # Simple OLS slope for y = a + b*x
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)
    if denom <= 0:
        return float("nan")
    return float(np.sum(x_centered * (y - y.mean())) / denom)


def load_points(results_json: Path) -> List[dict]:
    data = json.loads(results_json.read_text())
    comps = data.get("comparisons", [])
    needed = ["model", "brain_session", "correlation"]
    points = [c for c in comps if all(k in c for k in needed)]
    if not points:
        raise ValueError("No usable comparison entries found in results JSON.")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Test BabyLM token-growth vs brain-alignment trend")
    parser.add_argument("--results-json", required=True, help="Path to language_model_results.json")
    parser.add_argument(
        "--token-map",
        default=None,
        help="JSON dict model->token_count; if omitted, uses babylm stage number (5/7/9) as proxy",
    )
    args = parser.parse_args()

    points = load_points(Path(args.results_json))

    models = sorted({p["model"] for p in points})
    sessions = sorted({p["brain_session"] for p in points})
    token_map = parse_token_map(args.token_map, models)

    by_session: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for p in points:
        m = p["model"]
        if m not in token_map:
            continue
        tokens = token_map[m]
        z = fisher_z(float(p["correlation"]))
        by_session[p["brain_session"]].append((tokens, z, m))

    # Keep only sessions with >=3 models for stable slope estimate.
    usable_sessions = [s for s in sessions if len(by_session[s]) >= 3]
    if not usable_sessions:
        raise ValueError("No session has at least 3 models. Cannot estimate trend.")

    print("=== Per-session slopes (Fisher-z alignment vs log(tokens)) ===")
    observed_slopes = []
    for s in usable_sessions:
        rows = by_session[s]
        x = np.log(np.array([r[0] for r in rows], dtype=float))
        y = np.array([r[1] for r in rows], dtype=float)
        b = slope(x, y)
        observed_slopes.append(b)
        print(f"{s}: slope={b:.6f} ({'decrease' if b < 0 else 'increase'})")

    observed_mean_slope = float(np.mean(observed_slopes))
    print(f"\nObserved mean slope across sessions: {observed_mean_slope:.6f}")

    # Exact permutation test:
    # permute token labels within each session, compute mean slope each time.
    # One-sided p-value for hypothesis slope < 0.
    session_perms = {}
    for s in usable_sessions:
        rows = by_session[s]
        tokens = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)
        unique_perm_slopes = []
        for perm in itertools.permutations(tokens):
            x = np.log(np.array(perm, dtype=float))
            unique_perm_slopes.append(slope(x, y))
        session_perms[s] = np.array(unique_perm_slopes, dtype=float)

    all_perm_means = []
    for combo in itertools.product(*(session_perms[s] for s in usable_sessions)):
        all_perm_means.append(float(np.mean(combo)))
    all_perm_means = np.array(all_perm_means, dtype=float)

    p_one_sided = float(np.mean(all_perm_means <= observed_mean_slope))
    p_two_sided = float(np.mean(np.abs(all_perm_means) >= abs(observed_mean_slope)))

    print("\n=== Permutation test (exact, within-session shuffles) ===")
    print(f"Permutations evaluated: {len(all_perm_means)}")
    print(f"p(one-sided, decrease): {p_one_sided:.6f}")
    print(f"p(two-sided): {p_two_sided:.6f}")

    print("\nInterpretation:")
    if observed_mean_slope < 0 and p_one_sided < 0.05:
        print("Supported: alignment decreases as token exposure increases.")
    elif observed_mean_slope < 0:
        print("Direction matches hypothesis, but evidence is weak with current sample size.")
    else:
        print("Current data does not support a decreasing trend.")


if __name__ == "__main__":
    main()
