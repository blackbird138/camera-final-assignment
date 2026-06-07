#!/usr/bin/env python3
"""Recolor saved depth .npy outputs into official-style demo PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


CV2_COLORMAPS = {
    "inferno": "COLORMAP_INFERNO",
    "jet": "COLORMAP_JET",
    "turbo": "COLORMAP_TURBO",
    "magma": "COLORMAP_MAGMA",
    "plasma": "COLORMAP_PLASMA",
    "viridis": "COLORMAP_VIRIDIS",
    "cividis": "COLORMAP_CIVIDIS",
    "rainbow": "COLORMAP_RAINBOW",
    "coolwarm": "COLORMAP_COOL",
}

MPL_COLORMAPS = (
    "Spectral",
    "Spectral_r",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "turbo",
)

STYLE_CHOICES = ("auto", "da2-official", "da3-official", "simple")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs",
        default=Path("outputs"),
        type=Path,
        help="Root outputs directory containing da2-* and da3-* folders.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=("diw", "eth3d"),
        help="Top-level output directories to skip.",
    )
    parser.add_argument(
        "--style",
        default="auto",
        choices=STYLE_CHOICES,
        help="Visualization style. auto uses da2-official for da2-* and da3-official for da3-*.",
    )
    parser.add_argument(
        "--colormap",
        default=None,
        help="Override colormap. Official defaults are da2=Spectral_r and da3=Spectral.",
    )
    parser.add_argument(
        "--vis-larger-is",
        choices=("auto", "closer", "farther"),
        default="auto",
        help="Meaning of larger raw depth values. auto uses summary.json, then da2=closer and da3=farther.",
    )
    parser.add_argument(
        "--invert-vis",
        choices=("auto", "true", "false"),
        default="auto",
        help="Whether to invert visualization. auto uses summary.json, otherwise false.",
    )
    parser.add_argument(
        "--no-preserve-layout",
        action="store_true",
        help="Do not preserve DA2 side-by-side original+depth PNG layout; write depth-only PNGs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files already in recolor directories.")
    return parser.parse_args()


def normalize_depth_for_vis(depth, np_module, larger_is: str, invert: bool):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    denom = depth_max - depth_min
    if denom < 1e-8:
        return np_module.zeros_like(depth, dtype=np_module.uint8)

    normalized = (depth - depth_min) / denom
    if larger_is == "farther":
        normalized = 1.0 - normalized
    if invert:
        normalized = 1.0 - normalized
    return (normalized * 255.0).astype(np_module.uint8)


def rgb_to_bgr(image):
    return image[..., ::-1]


def visualize_da2_official(depth, colormap: str):
    import matplotlib
    import numpy as np

    depth_min = float(depth.min())
    depth_max = float(depth.max())
    denom = depth_max - depth_min
    if denom < 1e-8:
        depth_uint8 = np.zeros_like(depth, dtype=np.uint8)
    else:
        normalized = (depth - depth_min) / denom
        depth_uint8 = (normalized * 255.0).astype(np.uint8)

    cm = matplotlib.colormaps[colormap]
    return rgb_to_bgr((cm(depth_uint8)[..., :3] * 255.0).astype(np.uint8))


def visualize_da3_official(depth, colormap: str, percentile: float = 2.0):
    import matplotlib
    import numpy as np

    depth = depth.astype(np.float32, copy=True)
    valid_mask = depth > 0
    depth[valid_mask] = 1.0 / depth[valid_mask]

    if valid_mask.sum() <= 10:
        depth_min = 0.0
        depth_max = 0.0
    else:
        valid_depth = depth[valid_mask]
        depth_min = float(np.percentile(valid_depth, percentile))
        depth_max = float(np.percentile(valid_depth, 100.0 - percentile))

    if depth_min == depth_max:
        depth_min -= 1e-6
        depth_max += 1e-6

    normalized = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    normalized = 1.0 - normalized
    cm = matplotlib.colormaps[colormap]
    return rgb_to_bgr((cm(normalized)[..., :3] * 255.0).astype(np.uint8))


def resolve_style(output_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested

    name = output_dir.name.lower()
    if name.startswith("da3"):
        return "da3-official"
    if name.startswith("da2"):
        return "da2-official"
    return "simple"


def resolve_colormap(style: str, requested: Optional[str]) -> str:
    if requested:
        return requested
    if style == "da3-official":
        return "Spectral"
    if style == "da2-official":
        return "Spectral_r"
    return "jet"


def colorize_depth(depth, *, style: str, colormap: str, larger_is: str, invert: bool):
    import cv2
    import numpy as np

    if style == "da2-official":
        return visualize_da2_official(depth, colormap)
    if style == "da3-official":
        return visualize_da3_official(depth, colormap)

    if colormap in CV2_COLORMAPS:
        depth_uint8 = normalize_depth_for_vis(depth, np, larger_is, invert)
        return cv2.applyColorMap(depth_uint8, getattr(cv2, CV2_COLORMAPS[colormap]))

    cm_input = normalize_depth_for_vis(depth, np, larger_is, invert).astype(np.float32) / 255.0
    import matplotlib

    return rgb_to_bgr((matplotlib.colormaps[colormap](cm_input)[..., :3] * 255.0).astype(np.uint8))


def load_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_larger_is(output_dir: Path, requested: str, summary: dict) -> str:
    if requested != "auto":
        return requested

    summary_value = summary.get("vis_larger_is")
    if summary_value in {"closer", "farther"}:
        return summary_value

    name = output_dir.name.lower()
    if name.startswith("da3"):
        return "farther"
    return "closer"


def resolve_invert(requested: str, summary: dict) -> bool:
    if requested == "true":
        return True
    if requested == "false":
        return False
    return bool(summary.get("invert_vis", False))


def compose_with_existing_layout(existing_png: Path, depth_vis, cv2_module, np_module):
    existing = cv2_module.imread(str(existing_png), cv2_module.IMREAD_COLOR)
    if existing is None:
        return depth_vis

    depth_h, depth_w = depth_vis.shape[:2]
    existing_h, existing_w = existing.shape[:2]
    if existing_h != depth_h or existing_w < depth_w * 2 + 40:
        return depth_vis

    split_w = 50
    left_w = existing_w - depth_w - split_w
    if left_w <= 0:
        return depth_vis

    left = existing[:, :left_w]
    split = np_module.ones((existing_h, split_w, 3), dtype=np_module.uint8) * 255
    return cv2_module.hconcat([left, split, depth_vis])


def recolor_directory(
    output_dir: Path,
    *,
    requested_style: str,
    colormap: str,
    requested_larger_is: str,
    requested_invert: str,
    preserve_layout: bool,
    overwrite: bool,
) -> tuple[int, int]:
    import cv2
    import numpy as np

    summary = load_summary(output_dir)
    style = resolve_style(output_dir, requested_style)
    colormap = resolve_colormap(style, colormap)
    larger_is = resolve_larger_is(output_dir, requested_larger_is, summary)
    invert = resolve_invert(requested_invert, summary)

    recolor_dir = output_dir / "recolor"
    written = 0
    skipped = 0

    for npy_path in sorted(output_dir.rglob("*.npy")):
        if recolor_dir in npy_path.parents:
            continue

        relative = npy_path.relative_to(output_dir).with_suffix(".png")
        output_png = recolor_dir / relative
        if output_png.exists() and not overwrite:
            skipped += 1
            continue

        depth = np.load(str(npy_path))
        depth_vis = colorize_depth(depth, style=style, colormap=colormap, larger_is=larger_is, invert=invert)

        existing_png = npy_path.with_suffix(".png")
        if preserve_layout and existing_png.exists():
            depth_vis = compose_with_existing_layout(existing_png, depth_vis, cv2, np)

        output_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_png), depth_vis)
        written += 1

    metadata = {
        "source_dir": str(output_dir),
        "style": style,
        "colormap": colormap,
        "vis_larger_is": larger_is,
        "invert_vis": invert,
        "preserve_layout": preserve_layout,
        "written": written,
        "skipped_existing": skipped,
    }
    recolor_dir.mkdir(parents=True, exist_ok=True)
    with (recolor_dir / "recolor_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return written, skipped


def main() -> int:
    args = parse_args()
    outputs = args.outputs.resolve()
    if not outputs.is_dir():
        raise FileNotFoundError(f"Outputs directory not found: {outputs}")

    excluded = set(args.exclude)
    total_written = 0
    total_skipped = 0

    for output_dir in sorted(path for path in outputs.iterdir() if path.is_dir()):
        if output_dir.name in excluded or output_dir.name == "recolor":
            continue

        written, skipped = recolor_directory(
            output_dir,
            requested_style=args.style,
            colormap=args.colormap,
            requested_larger_is=args.vis_larger_is,
            requested_invert=args.invert_vis,
            preserve_layout=not args.no_preserve_layout,
            overwrite=args.overwrite,
        )
        total_written += written
        total_skipped += skipped
        print(f"{output_dir.name}: wrote {written}, skipped {skipped}")

    print(f"Done: wrote {total_written}, skipped {total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
