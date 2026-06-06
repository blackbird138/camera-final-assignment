#!/usr/bin/env python3
"""Evaluate dense depth predictions against ground-truth depth maps."""

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
    transform: str


def parse_model_spec(value: str) -> ModelSpec:
    parts = value.split("=")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("Model spec must be name=outdir or name=outdir=transform")
    transform = parts[2] if len(parts) == 3 else "identity"
    if transform not in {"identity", "negate", "inverse", "neg_inverse", "auto"}:
        raise argparse.ArgumentTypeError("transform must be identity, negate, inverse, neg_inverse, or auto")
    return ModelSpec(name=parts[0], outdir=Path(parts[1]), transform=transform)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root used by the pairs CSV.")
    parser.add_argument("--pairs", required=True, type=Path, help="CSV with image, gt_depth, width, height columns.")
    parser.add_argument("--model", required=True, action="append", type=parse_model_spec)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--fit-sample-pixels", default=200000, type=int, help="Max valid pixels for scale/shift fit.")
    parser.add_argument("--eval-sample-pixels", default=1000000, type=int, help="Max valid pixels for metrics per image; <=0 uses all.")
    parser.add_argument("--seed", default=20260606, type=int)
    parser.add_argument("--min-depth", default=1e-6, type=float)
    parser.add_argument("--max-depth", default=math.inf, type=float)
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


def candidate_keys(path_text: str) -> list[str]:
    path = Path(path_text)
    candidates = [path.as_posix().lstrip("./"), path.name]
    parts = path.parts
    for anchor in ("data", "raw", "eth3d", "diw"):
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


def find_depth_prediction(model_outdir: Path, image_key: str, runtime_index: dict[str, dict[str, str]]) -> Path:
    for key in candidate_keys(image_key):
        row = runtime_index.get(key)
        if row:
            output = Path(row["output_npy"])
            if output.exists():
                return output
            candidate = model_outdir / output
            if candidate.exists():
                return candidate

    image_path = Path(image_key)
    candidates = [
        model_outdir / image_path.with_suffix(".npy"),
        model_outdir / Path(image_path.name).with_suffix(".npy"),
    ]
    if len(image_path.parts) > 1:
        candidates.append(model_outdir / Path(*image_path.parts[1:]).with_suffix(".npy"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction .npy for {image_key} under {model_outdir}")


def read_pfm(path: Path):
    import numpy as np

    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Not a PFM file: {path}")
        dims = handle.readline().decode("ascii").strip()
        while dims.startswith("#"):
            dims = handle.readline().decode("ascii").strip()
        width, height = map(int, dims.split())
        scale = float(handle.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(handle, endian + "f", width * height * channels)
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def load_gt_depth(path: Path, width: int, height: int):
    import numpy as np
    from PIL import Image

    suffix = path.suffix.lower()
    size = path.stat().st_size
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data else data.files[0]
        return data[key].astype(np.float32)
    if suffix == ".pfm":
        return read_pfm(path).astype(np.float32)
    if size == width * height * 4:
        return np.fromfile(path, dtype="<f4").reshape((height, width)).astype(np.float32)
    with Image.open(path) as image:
        arr = np.asarray(image)
    return arr.astype(np.float32)


def load_pred_depth(path: Path):
    import numpy as np

    pred = np.load(path)
    if pred.ndim == 3:
        pred = pred.squeeze()
    return pred.astype(np.float32)


def resize_to(depth, width: int, height: int):
    import numpy as np
    from PIL import Image

    if depth.shape[:2] == (height, width):
        return depth.astype(np.float32)
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    resized = image.resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def transform_prediction(pred, transform: str):
    import numpy as np

    eps = 1e-6
    if transform == "identity":
        return pred
    if transform == "negate":
        return -pred
    if transform == "inverse":
        return 1.0 / np.maximum(pred, eps)
    if transform == "neg_inverse":
        return -1.0 / np.maximum(pred, eps)
    raise ValueError(f"Unsupported transform: {transform}")


def sample_indices(count: int, limit: int, seed: int):
    import numpy as np

    if limit <= 0 or count <= limit:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return rng.choice(count, size=limit, replace=False)


def fit_scale_shift(pred, gt):
    import numpy as np

    a_00 = float(np.sum(pred * pred))
    a_01 = float(np.sum(pred))
    a_11 = float(pred.size)
    b_0 = float(np.sum(pred * gt))
    b_1 = float(np.sum(gt))
    det = a_00 * a_11 - a_01 * a_01
    if abs(det) < 1e-8:
        return 1.0, 0.0
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    return scale, shift


def compute_metrics(pred, gt, min_depth: float):
    import numpy as np

    pred = np.maximum(pred, min_depth)
    gt = np.maximum(gt, min_depth)
    ratio = np.maximum(pred / gt, gt / pred)
    diff = pred - gt
    log_diff = np.log(pred) - np.log(gt)
    return {
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "absrel": float(np.mean(np.abs(diff) / gt)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "log_rmse": float(np.sqrt(np.mean(log_diff * log_diff))),
        "silog": float(np.sqrt(max(np.mean(log_diff * log_diff) - np.mean(log_diff) ** 2, 0.0)) * 100.0),
    }


def evaluate_one(pred_raw, gt, valid_mask, transform: str, args: argparse.Namespace, row_index: int):
    import numpy as np

    valid_gt = gt[valid_mask]
    pred_candidates = ["identity", "negate", "inverse", "neg_inverse"] if transform == "auto" else [transform]
    best = None
    for candidate in pred_candidates:
        pred_t = transform_prediction(pred_raw, candidate)
        valid_pred = pred_t[valid_mask]
        finite = np.isfinite(valid_pred) & np.isfinite(valid_gt)
        valid_pred = valid_pred[finite]
        valid_gt_fit = valid_gt[finite]
        if valid_pred.size < 100:
            continue
        fit_idx = sample_indices(valid_pred.size, args.fit_sample_pixels, args.seed + row_index)
        scale, shift = fit_scale_shift(valid_pred[fit_idx], valid_gt_fit[fit_idx])
        aligned = valid_pred * scale + shift
        positive = aligned > args.min_depth
        aligned = aligned[positive]
        gt_eval = valid_gt_fit[positive]
        if aligned.size == 0:
            continue
        eval_idx = sample_indices(aligned.size, args.eval_sample_pixels, args.seed + row_index + 100000)
        metrics = compute_metrics(aligned[eval_idx], gt_eval[eval_idx], args.min_depth)
        metrics.update({"transform": candidate, "scale": scale, "shift": shift, "valid_pixels": int(aligned.size)})
        if best is None or metrics["delta1"] > best["delta1"]:
            best = metrics
    if best is None:
        raise RuntimeError("No valid pixels after alignment.")
    return best


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def main() -> int:
    import numpy as np

    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.pairs)
    detail_rows = []
    summary_rows = []

    for model in args.model:
        runtime_index = build_runtime_index(model.outdir)
        model_details = []
        for index, row in enumerate(rows):
            image_key = row["image"]
            width = int(row["width"])
            height = int(row["height"])
            gt_path = args.root / row["gt_depth"]
            pred_path = find_depth_prediction(model.outdir, image_key, runtime_index)
            gt = load_gt_depth(gt_path, width, height)
            pred = resize_to(load_pred_depth(pred_path), width, height)
            valid = np.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
            metrics = evaluate_one(pred, gt, valid, model.transform, args, index)
            detail = {
                "model": model.name,
                "scene": row.get("scene", ""),
                "image": image_key,
                "gt_depth": row["gt_depth"],
                "pred_depth": str(pred_path),
                **{key: (f"{value:.8f}" if isinstance(value, float) else value) for key, value in metrics.items()},
            }
            detail_rows.append(detail)
            model_details.append(metrics)

        summary = {
            "model": model.name,
            "outdir": str(model.outdir),
            "image_count": len(model_details),
            "mean_delta1": mean([m["delta1"] for m in model_details]),
            "mean_delta2": mean([m["delta2"] for m in model_details]),
            "mean_delta3": mean([m["delta3"] for m in model_details]),
            "mean_absrel": mean([m["absrel"] for m in model_details]),
            "mean_rmse": mean([m["rmse"] for m in model_details]),
            "mean_log_rmse": mean([m["log_rmse"] for m in model_details]),
            "mean_silog": mean([m["silog"] for m in model_details]),
            "median_valid_pixels": float(np.median([m["valid_pixels"] for m in model_details])) if model_details else math.nan,
            "requested_transform": model.transform,
        }
        summary_rows.append({key: (f"{value:.8f}" if isinstance(value, float) else value) for key, value in summary.items()})

    detail_fields = [
        "model",
        "scene",
        "image",
        "gt_depth",
        "pred_depth",
        "transform",
        "scale",
        "shift",
        "valid_pixels",
        "delta1",
        "delta2",
        "delta3",
        "absrel",
        "rmse",
        "log_rmse",
        "silog",
    ]
    summary_fields = [
        "model",
        "outdir",
        "image_count",
        "requested_transform",
        "mean_delta1",
        "mean_delta2",
        "mean_delta3",
        "mean_absrel",
        "mean_rmse",
        "mean_log_rmse",
        "mean_silog",
        "median_valid_pixels",
    ]
    write_csv(args.outdir / "dense_depth_results.csv", detail_rows, detail_fields)
    write_csv(args.outdir / "dense_depth_summary.csv", summary_rows, summary_fields)
    with (args.outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"pairs": str(args.pairs), "root": str(args.root), "models": summary_rows}, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {args.outdir / 'dense_depth_summary.csv'}")
    print(f"Wrote {args.outdir / 'dense_depth_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
