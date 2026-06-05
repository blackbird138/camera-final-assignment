#!/usr/bin/env python3
"""Run Depth Anything 3 inference and record per-image runtime."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".thumb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Path to the Depth-Anything-3 repo.")
    parser.add_argument(
        "--model-dir",
        default="depth-anything/DA3MONO-LARGE",
        help="Hugging Face repo id or local model directory.",
    )
    parser.add_argument("--img-path", required=True, type=Path, help="Image file or directory.")
    parser.add_argument("--image-list", type=Path, help="Optional text file with one image path per line.")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory for depth outputs and runtime CSV.")
    parser.add_argument("--process-res", default=2048, type=int, help="DA3 processing resolution.")
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=("upper_bound_resize", "lower_bound_resize"),
        help="DA3 resize strategy.",
    )
    parser.add_argument("--warmup", default=1, type=int, help="Warmup runs on the first image before timing.")
    parser.add_argument("--recursive", action="store_true", help="Search image directories recursively.")
    parser.add_argument("--save-npy", action="store_true", help="Save raw depth arrays as .npy files.")
    parser.add_argument(
        "--vis-larger-is",
        default="farther",
        choices=("closer", "farther"),
        help="Meaning of larger raw depth values for PNG visualization. DA3 defaults to farther.",
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
    return sorted(
        path
        for path in img_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS
    )


def relative_output_path(image: Path, root: Path, outdir: Path) -> Path:
    if root.is_dir():
        try:
            rel = image.relative_to(root)
        except ValueError:
            rel = Path(image.name)
    else:
        rel = Path(image.name)
    return outdir / rel.with_suffix(".png")


def normalize_depth_for_vis(depth, np_module, larger_is: str, invert: bool):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    denom = depth_max - depth_min
    if denom < 1e-8:
        depth_uint8 = np_module.zeros_like(depth, dtype=np_module.uint8)
    else:
        normalized = (depth - depth_min) / denom
        if larger_is == "farther":
            normalized = 1.0 - normalized
        if invert:
            normalized = 1.0 - normalized
        depth_uint8 = (normalized * 255.0).astype(np_module.uint8)
    return depth_uint8, depth_min, depth_max


def sync_if_needed(torch_module, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    img_path = args.img_path.resolve()
    outdir = args.outdir.resolve()

    if not repo.exists():
        raise FileNotFoundError(f"Depth-Anything-3 repo not found: {repo}")
    sys.path.insert(0, str(repo / "src"))

    import cv2
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

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

    model = DepthAnything3.from_pretrained(args.model_dir).to(device)

    for _ in range(max(args.warmup, 0)):
        _ = model.inference(
            image=[str(images[0])],
            process_res=args.process_res,
            process_res_method=args.process_res_method,
        )
        sync_if_needed(torch, device)

    rows = []
    start_all = time.perf_counter()

    for index, image in enumerate(images, start=1):
        sync_if_needed(torch, device)
        start = time.perf_counter()
        prediction = model.inference(
            image=[str(image)],
            process_res=args.process_res,
            process_res_method=args.process_res_method,
        )
        sync_if_needed(torch, device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        depth = prediction.depth[0]
        if hasattr(depth, "detach"):
            depth = depth.detach().cpu().numpy()
        depth_uint8, depth_min, depth_max = normalize_depth_for_vis(
            depth,
            np,
            args.vis_larger_is,
            args.invert_vis,
        )
        depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

        output_png = relative_output_path(image, img_path, outdir)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_png), depth_vis)

        output_npy = ""
        if args.save_npy:
            output_npy_path = output_png.with_suffix(".npy")
            np.save(str(output_npy_path), depth)
            output_npy = str(output_npy_path)

        height, width = depth.shape[:2]
        row = {
            "index": index,
            "image": str(image),
            "width": width,
            "height": height,
            "model_dir": args.model_dir,
            "process_res": args.process_res,
            "process_res_method": args.process_res_method,
            "device": device,
            "elapsed_ms": f"{elapsed_ms:.3f}",
            "depth_min": f"{depth_min:.6f}",
            "depth_max": f"{depth_max:.6f}",
            "vis_larger_is": args.vis_larger_is,
            "invert_vis": str(bool(args.invert_vis)),
            "output_png": str(output_png),
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
        "model_dir",
        "process_res",
        "process_res_method",
        "device",
        "elapsed_ms",
        "depth_min",
        "depth_max",
        "vis_larger_is",
        "invert_vis",
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
        "model_dir": args.model_dir,
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "device": device,
        "warmup": args.warmup,
        "vis_larger_is": args.vis_larger_is,
        "invert_vis": bool(args.invert_vis),
        "visualization": "near_bright_far_dark" if not args.invert_vis else "near_dark_far_bright",
        "total_ms": round(total_ms, 3),
        "mean_ms": round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else None,
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
