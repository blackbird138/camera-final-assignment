#!/usr/bin/env python3
"""Run Depth Anything V2 inference and record per-image runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".thumb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Path to the Depth-Anything-V2 repo.")
    parser.add_argument("--img-path", required=True, type=Path, help="Image file or directory.")
    parser.add_argument("--image-list", type=Path, help="Optional text file with one image path per line.")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory for depth outputs and runtime CSV.")
    parser.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS), help="Model encoder.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path. Defaults to repo/checkpoints/depth_anything_v2_{encoder}.pth.")
    parser.add_argument("--input-size", default=518, type=int, help="Depth Anything V2 input size.")
    parser.add_argument("--warmup", default=1, type=int, help="Warmup runs on the first image before timing.")
    parser.add_argument("--grayscale", action="store_true", help="Save grayscale depth instead of Spectral_r colormap.")
    parser.add_argument("--pred-only", action="store_true", help="Save only prediction instead of original + prediction comparison.")
    parser.add_argument("--save-npy", action="store_true", help="Also save raw depth arrays as .npy files.")
    parser.add_argument(
        "--npy-dtype",
        default="float32",
        choices=("float32", "float16"),
        help="Dtype for saved .npy depth arrays. Use float16 for large benchmarks to save space.",
    )
    parser.add_argument("--skip-png", action="store_true", help="Skip PNG visualization output for large benchmarks.")
    parser.add_argument("--recursive", action="store_true", help="Search image directories recursively.")
    parser.add_argument(
        "--vis-larger-is",
        default="closer",
        choices=("closer", "farther"),
        help="Meaning of larger raw depth values for PNG visualization. DA2 defaults to closer.",
    )
    parser.add_argument(
        "--invert-vis",
        action="store_true",
        help="Invert final PNG visualization after applying --vis-larger-is.",
    )
    return parser.parse_args()


def find_images(img_path: Path, recursive: bool, image_list: Path | None = None) -> list[Path]:
    if image_list is not None:
        root = img_path if img_path.is_dir() else img_path.parent
        images = []
        with image_list.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = root / path
                if not path.exists():
                    raise FileNotFoundError(f"Image from list does not exist: {path}")
                images.append(path)
        return images

    if img_path.is_file():
        return [img_path]

    if not img_path.is_dir():
        raise FileNotFoundError(f"Image path does not exist: {img_path}")

    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in img_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS
    ]
    return sorted(images)


def sync_if_needed(torch_module, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def normalize_depth_for_vis(depth, np_module, larger_is: str, invert: bool):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    denom = depth_max - depth_min
    if denom < 1e-8:
        return np_module.zeros_like(depth, dtype=np_module.uint8), depth_min, depth_max
    normalized = (depth - depth_min) / denom
    if larger_is == "farther":
        normalized = 1.0 - normalized
    if invert:
        normalized = 1.0 - normalized
    depth_uint8 = (normalized * 255.0).astype(np_module.uint8)
    return depth_uint8, depth_min, depth_max


def apply_spectral_colormap(depth_uint8):
    import matplotlib

    depth_vis = matplotlib.colormaps["Spectral_r"](depth_uint8)[..., :3]
    return (depth_vis * 255.0).astype("uint8")[..., ::-1]


def relative_output_path(image: Path, root: Path, outdir: Path) -> Path:
    if root.is_dir():
        try:
            rel = image.relative_to(root)
        except ValueError:
            rel = Path(image.name)
    else:
        rel = Path(image.name)
    return outdir / rel.with_suffix(".png")


def read_image_bgr(path: Path):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    img_path = args.img_path.resolve()
    outdir = args.outdir.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else repo / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth"

    if not repo.exists():
        raise FileNotFoundError(f"Depth-Anything-V2 repo not found: {repo}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    sys.path.insert(0, str(repo))

    import cv2
    import numpy as np
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    images = find_images(img_path, args.recursive, args.image_list)
    if not images:
        raise RuntimeError(f"No images found under: {img_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    first_img = read_image_bgr(images[0])

    with torch.inference_mode():
        for _ in range(max(args.warmup, 0)):
            _ = model.infer_image(first_img, args.input_size)
            sync_if_needed(torch, device)

    rows = []
    start_all = time.perf_counter()

    with torch.inference_mode():
        for index, image in enumerate(images, start=1):
            load_start = time.perf_counter()
            raw_img = read_image_bgr(image)
            load_ms = (time.perf_counter() - load_start) * 1000.0

            sync_if_needed(torch, device)
            start = time.perf_counter()
            depth = model.infer_image(raw_img, args.input_size)
            sync_if_needed(torch, device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            depth_min = float(depth.min())
            depth_max = float(depth.max())

            output_png = relative_output_path(image, img_path, outdir)
            output_png.parent.mkdir(parents=True, exist_ok=True)
            output_png_text = ""

            if not args.skip_png:
                depth_uint8, depth_min, depth_max = normalize_depth_for_vis(
                    depth,
                    np,
                    args.vis_larger_is,
                    args.invert_vis,
                )
                if args.grayscale:
                    depth_vis = depth_uint8
                else:
                    depth_vis = apply_spectral_colormap(depth_uint8)

                if args.pred_only:
                    saved = depth_vis
                else:
                    if depth_vis.ndim == 2:
                        depth_for_stack = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                    else:
                        depth_for_stack = depth_vis
                    if depth_for_stack.shape[:2] != raw_img.shape[:2]:
                        depth_for_stack = cv2.resize(depth_for_stack, (raw_img.shape[1], raw_img.shape[0]))
                    split = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255
                    saved = cv2.hconcat([raw_img, split, depth_for_stack])

                cv2.imwrite(str(output_png), saved)
                output_png_text = str(output_png)

            output_npy = ""
            if args.save_npy:
                output_npy_path = output_png.with_suffix(".npy")
                np.save(str(output_npy_path), depth.astype(args.npy_dtype, copy=False))
                output_npy = str(output_npy_path)

            height, width = raw_img.shape[:2]
            row = {
                "index": index,
                "image": str(image),
                "width": width,
                "height": height,
                "encoder": args.encoder,
                "input_size": args.input_size,
                "device": device,
                "checkpoint": str(checkpoint),
                "load_ms": f"{load_ms:.3f}",
                "elapsed_ms": f"{elapsed_ms:.3f}",
                "depth_min": f"{depth_min:.6f}",
                "depth_max": f"{depth_max:.6f}",
                "vis_larger_is": args.vis_larger_is,
                "invert_vis": str(bool(args.invert_vis)),
                "skip_png": str(bool(args.skip_png)),
                "npy_dtype": args.npy_dtype if args.save_npy else "",
                "output_png": output_png_text,
                "output_npy": output_npy,
            }
            rows.append(row)
            print(f"[{index}/{len(images)}] {image.name}: {elapsed_ms:.2f} ms")

    total_ms = (time.perf_counter() - start_all) * 1000.0
    csv_path = outdir / "runtime.csv"
    fieldnames = [
        "index",
        "image",
        "width",
        "height",
        "encoder",
        "input_size",
        "device",
        "checkpoint",
        "load_ms",
        "elapsed_ms",
        "depth_min",
        "depth_max",
        "vis_larger_is",
        "invert_vis",
        "skip_png",
        "npy_dtype",
        "output_png",
        "output_npy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed_values = [float(row["elapsed_ms"]) for row in rows]
    summary = {
        "image_count": len(rows),
        "encoder": args.encoder,
        "input_size": args.input_size,
        "device": device,
        "checkpoint": str(checkpoint),
        "warmup": args.warmup,
        "vis_larger_is": args.vis_larger_is,
        "invert_vis": bool(args.invert_vis),
        "skip_png": bool(args.skip_png),
        "npy_dtype": args.npy_dtype if args.save_npy else None,
        "visualization": "grayscale" if args.grayscale else "da2_official_spectral_r",
        "total_ms": round(total_ms, 3),
        "mean_ms": round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else None,
        "median_ms": round(percentile(elapsed_values, 0.5), 3) if elapsed_values else None,
        "p95_ms": round(percentile(elapsed_values, 0.95), 3) if elapsed_values else None,
        "min_ms": round(min(elapsed_values), 3) if elapsed_values else None,
        "max_ms": round(max(elapsed_values), 3) if elapsed_values else None,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }
    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Runtime CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
