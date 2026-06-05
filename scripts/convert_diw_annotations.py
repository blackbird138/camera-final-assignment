#!/usr/bin/env python3
"""Convert official DIW CSV annotations to this project's point-pair CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Official DIW CSV, e.g. DIW_test.csv.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV in project point-pair format.")
    parser.add_argument(
        "--images-root",
        type=Path,
        help="Root directory for DIW images. Used to make relative image keys and optional file checks.",
    )
    parser.add_argument(
        "--path-mode",
        default="relative",
        choices=("relative", "basename", "as-is"),
        help="How to write image paths in the output CSV.",
    )
    parser.add_argument(
        "--image-prefix",
        default="",
        help="Optional prefix prepended to each output image key, such as DIW_test.",
    )
    parser.add_argument(
        "--coordinate-base",
        default=1,
        type=int,
        choices=(0, 1),
        help="Official DIW Torch CSV coordinates are 1-based; output coordinates are 0-based.",
    )
    parser.add_argument("--limit", type=int, help="Keep at most this many samples after optional shuffle.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before applying --limit.")
    parser.add_argument("--seed", default=20260605, type=int, help="Random seed for --shuffle.")
    parser.add_argument("--image-list", type=Path, help="Optional output text file with one image key per line.")
    parser.add_argument("--strict", action="store_true", help="Fail if an image key cannot be resolved under --images-root.")
    return parser.parse_args()


def read_official_diw_csv(path: Path) -> list[tuple[str, list[str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]

    if len(rows) % 2 != 0:
        raise ValueError(f"Expected alternating image/annotation rows, got odd row count: {len(rows)}")

    samples = []
    for index in range(0, len(rows), 2):
        image_row = rows[index]
        point_row = rows[index + 1]
        if len(point_row) < 5:
            raise ValueError(f"Annotation row must have at least 5 columns: {point_row}")
        samples.append((image_row[0].strip(), [cell.strip() for cell in point_row[:5]]))
    return samples


def normalize_path(path_text: str) -> str:
    return Path(path_text).as_posix().lstrip("./")


def resolve_image_key(path_text: str, images_root: Path | None, path_mode: str, image_prefix: str, strict: bool) -> str:
    source = Path(path_text)

    if path_mode == "as-is":
        key = normalize_path(path_text)
    elif path_mode == "basename":
        key = source.name
    else:
        if images_root is None:
            key = source.name
        else:
            root = images_root.resolve()
            candidates: list[Path] = []
            if source.is_absolute():
                candidates.append(source)
            else:
                candidates.append(root / source)
            candidates.append(root / source.name)

            existing = next((candidate for candidate in candidates if candidate.exists()), None)
            if existing is not None:
                try:
                    key = existing.resolve().relative_to(root).as_posix()
                except ValueError:
                    key = existing.name
            elif source.is_absolute():
                try:
                    key = source.resolve().relative_to(root).as_posix()
                except ValueError:
                    key = source.name
            else:
                key = normalize_path(path_text)

            if strict and not (root / key).exists():
                raise FileNotFoundError(f"Cannot resolve DIW image under {root}: {path_text} -> {key}")

    if image_prefix:
        key = f"{image_prefix.strip('/')}/{key}"
    return key


def convert_coord(value: str, coordinate_base: int) -> int:
    coord = int(float(value))
    if coordinate_base == 1:
        coord -= 1
    if coord < 0:
        raise ValueError(f"Coordinate became negative after base conversion: {value}")
    return coord


def main() -> int:
    args = parse_args()
    samples = read_official_diw_csv(args.input)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(samples)
    if args.limit is not None:
        samples = samples[: args.limit]

    output_rows = []
    image_keys = []
    for index, (image_path, point_row) in enumerate(samples, start=1):
        y_a, x_a, y_b, x_b, relation = point_row
        relation = relation[:1]
        if relation not in {">", "<"}:
            raise ValueError(f"DIW relation must be > or <: {point_row}")

        # DIW README: ">" means point A is further away than point B.
        closer = "B" if relation == ">" else "A"
        image_key = resolve_image_key(
            image_path,
            args.images_root,
            args.path_mode,
            args.image_prefix,
            args.strict,
        )
        image_keys.append(image_key)
        output_rows.append(
            {
                "image": image_key,
                "pair_id": f"diw_{index:06d}",
                "ax": convert_coord(x_a, args.coordinate_base),
                "ay": convert_coord(y_a, args.coordinate_base),
                "bx": convert_coord(x_b, args.coordinate_base),
                "by": convert_coord(y_b, args.coordinate_base),
                "closer": closer,
                "notes": f"diw_relation={relation}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "pair_id", "ax", "ay", "bx", "by", "closer", "notes"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    if args.image_list:
        args.image_list.parent.mkdir(parents=True, exist_ok=True)
        unique_keys = list(dict.fromkeys(image_keys))
        with args.image_list.open("w", encoding="utf-8") as handle:
            for key in unique_keys:
                handle.write(f"{key}\n")

    print(f"Wrote {len(output_rows)} DIW point pairs to {args.output}")
    if args.image_list:
        print(f"Wrote {len(set(image_keys))} image keys to {args.image_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
