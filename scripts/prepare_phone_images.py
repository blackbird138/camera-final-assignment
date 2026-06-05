#!/usr/bin/env python3
"""Normalize phone photos for model inference.

Converts HEIC/HEIF/JPEG/PNG inputs to orientation-corrected JPEG files and
writes a mapping CSV. Keep the original files under data/raw.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


INPUT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with raw phone photos.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for normalized JPEG files.")
    parser.add_argument("--quality", default=95, type=int, help="JPEG quality.")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in INPUT_EXTENSIONS
    )


def output_path_for(output_dir: Path, input_path: Path) -> Path:
    return output_dir / f"{input_path.stem}.jpg"


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise SystemExit("Please install Pillow first: python -m pip install pillow") from exc

    has_heic = any(path.suffix.lower() in {".heic", ".heif"} for path in find_images(args.input_dir, args.recursive))
    if has_heic:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:
            raise SystemExit(
                "HEIC/HEIF input detected. Install support first: "
                "python -m pip install pillow-heif"
            ) from exc
        register_heif_opener()

    rows = []
    for input_path in find_images(args.input_dir, args.recursive):
        output_path = output_path_for(args.output_dir, input_path)
        if output_path.exists() and not args.overwrite:
            status = "skipped_exists"
            width = ""
            height = ""
        else:
            with Image.open(input_path) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                elif image.mode == "L":
                    image = image.convert("RGB")
                width, height = image.size
                image.save(output_path, format="JPEG", quality=args.quality, optimize=True)
            status = "converted"

        rows.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "status": status,
                "width": width,
                "height": height,
                "input_ext": input_path.suffix.lower(),
            }
        )

    mapping_path = args.output_dir / "mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input", "output", "status", "width", "height", "input_ext"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} input images")
    print(f"Wrote normalized images to {args.output_dir}")
    print(f"Wrote mapping CSV to {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
