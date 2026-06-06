#!/usr/bin/env python3
"""Build an image/depth CSV for ETH3D high-res training scenes."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="ETH3D root containing scene folders.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    parser.add_argument("--image-list", type=Path, help="Optional text file for inference scripts.")
    parser.add_argument("--scenes", nargs="*", help="Optional scene names to include.")
    parser.add_argument("--limit", type=int, help="Limit rows after optional shuffle.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows before applying --limit.")
    parser.add_argument("--seed", default=20260606, type=int, help="Random seed for --shuffle.")
    return parser.parse_args()


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def candidate_depth_paths(image: Path, root: Path) -> list[Path]:
    rel = image.resolve().relative_to(root.resolve())
    parts = list(rel.parts)
    candidates: list[Path] = []
    for marker in ("dslr_images", "images"):
        if marker in parts:
            idx = parts.index(marker)
            replaced = parts[:]
            replaced[idx] = "dslr_depth"
            candidates.append(root / Path(*replaced))
            replaced = parts[:]
            replaced[idx] = "depth"
            candidates.append(root / Path(*replaced))
            replaced = parts[:]
            replaced[idx] = "ground_truth_depth"
            candidates.append(root / Path(*replaced))
    if len(parts) >= 2:
        scene = parts[0]
        name = parts[-1]
        candidates.extend(
            [
                root / scene / "dslr_depth" / name,
                root / scene / "depth" / name,
                root / scene / "ground_truth_depth" / name,
            ]
        )
    return list(dict.fromkeys(candidates))


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    scene_filter = set(args.scenes or [])
    rows = []

    for image in sorted(root.rglob("*")):
        if not image.is_file() or image.suffix not in IMAGE_EXTENSIONS:
            continue
        rel_parts = image.resolve().relative_to(root).parts
        if "dslr_images" not in rel_parts and "images" not in rel_parts:
            continue
        scene = rel_parts[0] if rel_parts else ""
        if scene_filter and scene not in scene_filter:
            continue
        depth = next((path for path in candidate_depth_paths(image, root) if path.exists()), None)
        if depth is None:
            continue
        width, height = image_size(image)
        rows.append(
            {
                "scene": scene,
                "image": relative(image, root),
                "gt_depth": relative(depth, root),
                "width": width,
                "height": height,
            }
        )

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene", "image", "gt_depth", "width", "height"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.image_list:
        args.image_list.parent.mkdir(parents=True, exist_ok=True)
        with args.image_list.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(f"{row['image']}\n")

    print(f"Wrote {len(rows)} ETH3D pairs to {args.output}")
    if args.image_list:
        print(f"Wrote image list to {args.image_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
