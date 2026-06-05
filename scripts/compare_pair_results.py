#!/usr/bin/env python3
"""Compare two models on the same point-pair evaluation CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-results", required=True, type=Path, help="pair_results.csv from evaluate_depth_outputs.py.")
    parser.add_argument("--model-a", required=True, help="First model name in pair_results.csv.")
    parser.add_argument("--model-b", required=True, help="Second model name in pair_results.csv.")
    parser.add_argument("--outdir", type=Path, help="Optional directory for JSON/CSV summary.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["image"], row.get("pair_id", ""))


def log_binomial_pmf(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)


def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    smaller = min(a_only, b_only)
    log_terms = [log_binomial_pmf(discordant, k) for k in range(smaller + 1)]
    max_log = max(log_terms)
    tail = math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)
    return min(1.0, 2.0 * tail)


def paired_difference_ci(a_only: int, b_only: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    diff = (a_only - b_only) / total if total else 0.0
    second_moment = (a_only + b_only) / total if total else 0.0
    variance = max(second_moment - diff * diff, 0.0)
    se = math.sqrt(variance / total) if total else 0.0
    return diff, diff - z * se, diff + z * se


def main() -> int:
    args = parse_args()
    rows = read_csv(args.pair_results)

    by_model: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[pair_key(row)] = row

    if args.model_a not in by_model:
        raise ValueError(f"Model not found: {args.model_a}")
    if args.model_b not in by_model:
        raise ValueError(f"Model not found: {args.model_b}")

    a_rows = by_model[args.model_a]
    b_rows = by_model[args.model_b]
    common_keys = sorted(set(a_rows) & set(b_rows))
    if not common_keys:
        raise RuntimeError("No common point pairs found between the two models.")

    both_correct = 0
    both_wrong = 0
    a_only = 0
    b_only = 0
    details = []
    for key in common_keys:
        a_correct = a_rows[key].get("correct") == "1"
        b_correct = b_rows[key].get("correct") == "1"
        if a_correct and b_correct:
            both_correct += 1
            outcome = "both_correct"
        elif not a_correct and not b_correct:
            both_wrong += 1
            outcome = "both_wrong"
        elif a_correct:
            a_only += 1
            outcome = f"{args.model_a}_only"
        else:
            b_only += 1
            outcome = f"{args.model_b}_only"

        details.append(
            {
                "image": key[0],
                "pair_id": key[1],
                "closer": a_rows[key].get("closer", ""),
                f"{args.model_a}_prediction": a_rows[key].get("prediction", ""),
                f"{args.model_a}_correct": "1" if a_correct else "0",
                f"{args.model_b}_prediction": b_rows[key].get("prediction", ""),
                f"{args.model_b}_correct": "1" if b_correct else "0",
                "outcome": outcome,
            }
        )

    total = len(common_keys)
    a_correct_total = both_correct + a_only
    b_correct_total = both_correct + b_only
    diff, ci_low, ci_high = paired_difference_ci(a_only, b_only, total)
    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "common_pairs": total,
        "model_a_correct": a_correct_total,
        "model_b_correct": b_correct_total,
        "model_a_accuracy": a_correct_total / total,
        "model_b_accuracy": b_correct_total / total,
        "accuracy_diff_model_a_minus_model_b": diff,
        "accuracy_diff_95ci_low": ci_low,
        "accuracy_diff_95ci_high": ci_high,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "model_a_only_correct": a_only,
        "model_b_only_correct": b_only,
        "discordant_pairs": a_only + b_only,
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(a_only, b_only),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        with (args.outdir / "paired_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        fieldnames = [
            "image",
            "pair_id",
            "closer",
            f"{args.model_a}_prediction",
            f"{args.model_a}_correct",
            f"{args.model_b}_prediction",
            f"{args.model_b}_correct",
            "outcome",
        ]
        with (args.outdir / "paired_details.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(details)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
