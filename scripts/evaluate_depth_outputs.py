#!/usr/bin/env python3
"""Evaluate depth outputs without dense ground truth.

Main metric: manually annotated relative depth ordering accuracy.
Auxiliary metric: image-edge/depth-gradient alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    outdir: Path
    direction: str


def parse_model_spec(value: str) -> ModelSpec:
    parts = value.split("=")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "Model spec must be name=outdir or name=outdir=direction"
        )
    name, outdir = parts[0], Path(parts[1])
    direction = parts[2] if len(parts) == 3 else "auto"
    if direction not in {"auto", "larger_closer", "larger_farther"}:
        raise argparse.ArgumentTypeError(
            "direction must be auto, larger_closer, or larger_farther"
        )
    return ModelSpec(name=name, outdir=outdir, direction=direction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, type=Path, help="Root directory for original images.")
    parser.add_argument("--annotations", required=True, type=Path, help="CSV with point-pair annotations.")
    parser.add_argument(
        "--model",
        required=True,
        action="append",
        type=parse_model_spec,
        help="Model output spec: name=outdir or name=outdir=direction. Repeat for DA2/DA3.",
    )
    parser.add_argument("--outdir", required=True, type=Path, help="Directory for evaluation CSV/JSON outputs.")
    parser.add_argument("--sample-radius", default=3, type=int, help="Median sample radius in depth-map pixels.")
    parser.add_argument(
        "--tie-threshold",
        default=0.01,
        type=float,
        help="Prediction is tie if abs(depth_a-depth_b) is below this fraction of per-image depth range.",
    )
    parser.add_argument("--edge-metrics", action="store_true", help="Also compute image-edge/depth-gradient metrics.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_key(path_text: str) -> str:
    return Path(path_text).as_posix().lstrip("./")


def candidate_keys(path_text: str) -> list[str]:
    path = Path(path_text)
    parts = path.parts
    candidates = [normalize_key(path_text), path.name]
    for anchor in ("data", "raw", "examples", "phone_images"):
        if anchor in parts:
            idx = parts.index(anchor)
            candidates.append(Path(*parts[idx:]).as_posix())
            if idx + 1 < len(parts):
                candidates.append(Path(*parts[idx + 1 :]).as_posix())
    return list(dict.fromkeys(candidates))


def build_runtime_index(model_outdir: Path) -> dict[str, dict[str, str]]:
    runtime_csv = model_outdir / "runtime.csv"
    if not runtime_csv.exists():
        return {}
    index = {}
    for row in read_csv(runtime_csv):
        image = row.get("image", "")
        output_npy = row.get("output_npy", "")
        if not output_npy:
            continue
        for key in candidate_keys(image):
            index[key] = row
    return index


def find_depth_file(model_outdir: Path, image_key: str, runtime_index: dict[str, dict[str, str]]) -> Path:
    for key in candidate_keys(image_key):
        row = runtime_index.get(key)
        if row:
            output_npy = Path(row["output_npy"])
            if output_npy.exists():
                return output_npy
            candidate = model_outdir / output_npy
            if candidate.exists():
                return candidate

    image_path = Path(image_key)
    possible_rel = [
        image_path.with_suffix(".npy"),
        Path(image_path.name).with_suffix(".npy"),
    ]
    if len(image_path.parts) > 1:
        possible_rel.append(Path(*image_path.parts[1:]).with_suffix(".npy"))
    for rel in possible_rel:
        candidate = model_outdir / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No .npy depth output found for {image_key} under {model_outdir}")


def image_size(images_root: Path, image_key: str) -> tuple[int, int]:
    from PIL import Image

    path = images_root / image_key
    if not path.exists():
        path = images_root / Path(image_key).name
    with Image.open(path) as image:
        return image.size


def load_image_gray(images_root: Path, image_key: str):
    import numpy as np
    from PIL import Image

    path = images_root / image_key
    if not path.exists():
        path = images_root / Path(image_key).name
    with Image.open(path) as image:
        gray = image.convert("L")
        return np.asarray(gray, dtype=np.float32) / 255.0


def load_depth(path: Path):
    import numpy as np

    depth = np.load(path)
    if depth.ndim == 3:
        depth = depth.squeeze()
    return depth.astype(np.float32)


def sample_depth(depth, x: float, y: float, image_w: int, image_h: int, radius: int) -> float:
    import numpy as np

    depth_h, depth_w = depth.shape[:2]
    dx = int(round(x * (depth_w - 1) / max(image_w - 1, 1)))
    dy = int(round(y * (depth_h - 1) / max(image_h - 1, 1)))
    dx = min(max(dx, 0), depth_w - 1)
    dy = min(max(dy, 0), depth_h - 1)
    if radius <= 0:
        return float(depth[dy, dx])
    x0, x1 = max(dx - radius, 0), min(dx + radius + 1, depth_w)
    y0, y1 = max(dy - radius, 0), min(dy + radius + 1, depth_h)
    patch = depth[y0:y1, x0:x1]
    return float(np.median(patch))


def prediction_from_values(a_value: float, b_value: float, depth_range: float, direction: str, tie_threshold: float) -> str:
    margin = abs(a_value - b_value) / max(depth_range, 1e-8)
    if margin < tie_threshold:
        return "tie"
    if direction == "larger_closer":
        return "A" if a_value > b_value else "B"
    if direction == "larger_farther":
        return "A" if a_value < b_value else "B"
    raise ValueError(f"Unsupported direction: {direction}")


def score_rows(rows: list[dict], direction: str) -> dict[str, float | int]:
    total = 0
    correct = 0
    ties = 0
    margins = []
    for row in rows:
        gt = row["closer"]
        pred = row[f"pred_{direction}"]
        if gt not in {"A", "B"}:
            continue
        total += 1
        if pred == "tie":
            ties += 1
        elif pred == gt:
            correct += 1
        margins.append(float(row["relative_margin"]))
    accuracy = correct / total if total else 0.0
    non_tie_total = total - ties
    non_tie_accuracy = correct / non_tie_total if non_tie_total else 0.0
    mean_margin = sum(margins) / len(margins) if margins else 0.0
    return {
        "total_pairs": total,
        "correct": correct,
        "ties": ties,
        "accuracy": accuracy,
        "non_tie_accuracy": non_tie_accuracy,
        "mean_relative_margin": mean_margin,
    }


def summarize_runtime(model_outdir: Path) -> dict[str, float | int | str]:
    summary_json = model_outdir / "summary.json"
    if summary_json.exists():
        with summary_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            "runtime_image_count": data.get("image_count", ""),
            "runtime_mean_ms": data.get("mean_ms", ""),
            "runtime_min_ms": data.get("min_ms", ""),
            "runtime_max_ms": data.get("max_ms", ""),
        }
    runtime_csv = model_outdir / "runtime.csv"
    if not runtime_csv.exists():
        return {
            "runtime_image_count": "",
            "runtime_mean_ms": "",
            "runtime_min_ms": "",
            "runtime_max_ms": "",
        }
    values = [float(row["elapsed_ms"]) for row in read_csv(runtime_csv) if row.get("elapsed_ms")]
    return {
        "runtime_image_count": len(values),
        "runtime_mean_ms": sum(values) / len(values) if values else "",
        "runtime_min_ms": min(values) if values else "",
        "runtime_max_ms": max(values) if values else "",
    }


def resize_depth_to_image(depth, image_shape):
    import numpy as np
    from PIL import Image

    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    denom = max(depth_max - depth_min, 1e-8)
    depth_norm = ((depth - depth_min) / denom).astype(np.float32)
    image_h, image_w = image_shape
    pil = Image.fromarray(depth_norm)
    resized = pil.resize((image_w, image_h), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def gradient_magnitude(array):
    import numpy as np

    gy, gx = np.gradient(array.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def percentile_mask(values, percentile: float, greater: bool):
    import numpy as np

    threshold = np.percentile(values, percentile)
    if greater:
        return values >= threshold
    return values <= threshold


def compute_edge_metrics(images_root: Path, annotations: list[dict[str, str]], models: list[ModelSpec]):
    import numpy as np

    image_keys = sorted({row["image"] for row in annotations if row.get("image")})
    rows = []
    for model in models:
        runtime_index = build_runtime_index(model.outdir)
        for image_key in image_keys:
            try:
                gray = load_image_gray(images_root, image_key)
                depth_path = find_depth_file(model.outdir, image_key, runtime_index)
                depth = load_depth(depth_path)
            except FileNotFoundError:
                continue
            depth_resized = resize_depth_to_image(depth, gray.shape)
            image_grad = gradient_magnitude(gray)
            depth_grad = gradient_magnitude(depth_resized)
            edge_mask = percentile_mask(image_grad, 90.0, greater=True)
            flat_mask = percentile_mask(image_grad, 50.0, greater=False)
            edge_depth_mean = float(np.mean(depth_grad[edge_mask])) if np.any(edge_mask) else math.nan
            flat_depth_mean = float(np.mean(depth_grad[flat_mask])) if np.any(flat_mask) else math.nan
            ratio = edge_depth_mean / max(flat_depth_mean, 1e-8)
            rows.append(
                {
                    "model": model.name,
                    "image": image_key,
                    "edge_depth_grad_mean": f"{edge_depth_mean:.8f}",
                    "flat_depth_grad_mean": f"{flat_depth_mean:.8f}",
                    "edge_to_flat_ratio": f"{ratio:.4f}",
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    annotations = read_csv(args.annotations)
    for row in annotations:
        row["closer"] = row.get("closer", "").strip().upper()
        if row["closer"] not in {"A", "B"}:
            raise ValueError(f"closer must be A or B in row: {row}")

    pair_rows = []
    model_summaries = []

    for model in args.model:
        runtime_index = build_runtime_index(model.outdir)
        raw_rows = []
        for ann in annotations:
            image_key = normalize_key(ann["image"])
            depth_path = find_depth_file(model.outdir, image_key, runtime_index)
            depth = load_depth(depth_path)
            image_w, image_h = image_size(args.images_root, image_key)
            a_value = sample_depth(depth, float(ann["ax"]), float(ann["ay"]), image_w, image_h, args.sample_radius)
            b_value = sample_depth(depth, float(ann["bx"]), float(ann["by"]), image_w, image_h, args.sample_radius)
            depth_range = float(depth.max() - depth.min())
            relative_margin = abs(a_value - b_value) / max(depth_range, 1e-8)
            base = {
                "model": model.name,
                "image": image_key,
                "pair_id": ann.get("pair_id", ""),
                "ax": ann["ax"],
                "ay": ann["ay"],
                "bx": ann["bx"],
                "by": ann["by"],
                "closer": ann["closer"],
                "depth_a": f"{a_value:.8f}",
                "depth_b": f"{b_value:.8f}",
                "relative_margin": f"{relative_margin:.8f}",
                "depth_file": str(depth_path),
                "notes": ann.get("notes", ""),
            }
            for direction in ("larger_closer", "larger_farther"):
                base[f"pred_{direction}"] = prediction_from_values(
                    a_value, b_value, depth_range, direction, args.tie_threshold
                )
            raw_rows.append(base)

        if model.direction == "auto":
            larger_closer_score = score_rows(raw_rows, "larger_closer")
            larger_farther_score = score_rows(raw_rows, "larger_farther")
            selected_direction = (
                "larger_closer"
                if larger_closer_score["accuracy"] >= larger_farther_score["accuracy"]
                else "larger_farther"
            )
        else:
            selected_direction = model.direction

        selected_score = score_rows(raw_rows, selected_direction)
        runtime_summary = summarize_runtime(model.outdir)
        summary_row = {
            "model": model.name,
            "outdir": str(model.outdir),
            "selected_direction": selected_direction,
            "total_pairs": selected_score["total_pairs"],
            "correct": selected_score["correct"],
            "ties": selected_score["ties"],
            "accuracy": f"{selected_score['accuracy']:.4f}",
            "non_tie_accuracy": f"{selected_score['non_tie_accuracy']:.4f}",
            "mean_relative_margin": f"{selected_score['mean_relative_margin']:.6f}",
            **runtime_summary,
        }
        model_summaries.append(summary_row)

        for row in raw_rows:
            row["selected_direction"] = selected_direction
            row["prediction"] = row[f"pred_{selected_direction}"]
            row["correct"] = "1" if row["prediction"] == row["closer"] else "0"
            pair_rows.append(row)

    pair_fieldnames = [
        "model",
        "image",
        "pair_id",
        "ax",
        "ay",
        "bx",
        "by",
        "closer",
        "selected_direction",
        "prediction",
        "correct",
        "depth_a",
        "depth_b",
        "relative_margin",
        "pred_larger_closer",
        "pred_larger_farther",
        "depth_file",
        "notes",
    ]
    summary_fieldnames = [
        "model",
        "outdir",
        "selected_direction",
        "total_pairs",
        "correct",
        "ties",
        "accuracy",
        "non_tie_accuracy",
        "mean_relative_margin",
        "runtime_image_count",
        "runtime_mean_ms",
        "runtime_min_ms",
        "runtime_max_ms",
    ]
    write_csv(args.outdir / "pair_results.csv", pair_rows, pair_fieldnames)
    write_csv(args.outdir / "model_summary.csv", model_summaries, summary_fieldnames)

    output = {
        "annotations": str(args.annotations),
        "images_root": str(args.images_root),
        "sample_radius": args.sample_radius,
        "tie_threshold": args.tie_threshold,
        "models": model_summaries,
    }

    if args.edge_metrics:
        edge_rows = compute_edge_metrics(args.images_root, annotations, args.model)
        edge_fieldnames = [
            "model",
            "image",
            "edge_depth_grad_mean",
            "flat_depth_grad_mean",
            "edge_to_flat_ratio",
        ]
        write_csv(args.outdir / "edge_metrics.csv", edge_rows, edge_fieldnames)
        output["edge_metrics"] = edge_rows

    with (args.outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {args.outdir / 'model_summary.csv'}")
    print(f"Wrote {args.outdir / 'pair_results.csv'}")
    if args.edge_metrics:
        print(f"Wrote {args.outdir / 'edge_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

