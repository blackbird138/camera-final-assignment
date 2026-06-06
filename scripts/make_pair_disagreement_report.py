#!/usr/bin/env python3
"""Create an HTML report for pair-level model disagreements."""

from __future__ import annotations

import argparse
import csv
import html
import math
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-results", required=True, type=Path, help="pair_results.csv from evaluate_depth_outputs.py.")
    parser.add_argument("--images-root", required=True, type=Path, help="Root directory for original images.")
    parser.add_argument("--model-a", required=True, help="First model name in pair_results.csv.")
    parser.add_argument("--model-b", required=True, help="Second model name in pair_results.csv.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory for HTML and annotated images.")
    parser.add_argument("--limit-per-outcome", default=40, type=int, help="Maximum examples per disagreement outcome.")
    parser.add_argument(
        "--sort-by",
        default="wrong_margin",
        choices=("wrong_margin", "pair_id"),
        help="Sort disagreement examples by the wrong model's relative margin or by pair id.",
    )
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


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["image"], row.get("pair_id", ""))


def image_path(images_root: Path, image_key: str) -> Path:
    path = images_root / image_key
    if path.exists():
        return path
    fallback = images_root / Path(image_key).name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Image not found: {image_key}")


def depth_png_from_row(row: dict[str, str]) -> Path | None:
    depth_file = row.get("depth_file", "")
    if not depth_file:
        return None
    path = Path(depth_file).with_suffix(".png")
    return path if path.exists() else None


def rel_to(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def draw_marker(draw, x: float, y: float, label: str, color: tuple[int, int, int]) -> None:
    r = 7
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)
    draw.text((x + r + 3, y - r - 3), label, fill=color)


def make_annotated_image(row_a: dict[str, str], row_b: dict[str, str], images_root: Path, out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    source = image_path(images_root, row_a["image"])
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")

    max_w = 900
    scale = min(1.0, max_w / max(image.width, 1))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)

    draw = ImageDraw.Draw(image)
    ax = float(row_a["ax"]) * scale
    ay = float(row_a["ay"]) * scale
    bx = float(row_a["bx"]) * scale
    by = float(row_a["by"]) * scale
    draw.line((ax, ay, bx, by), fill=(255, 255, 255), width=2)
    draw_marker(draw, ax, ay, "A", (255, 60, 60))
    draw_marker(draw, bx, by, "B", (40, 160, 255))

    caption = (
        f"GT closer={row_a['closer']} | "
        f"{row_a['model']} pred={row_a['prediction']} correct={row_a['correct']} | "
        f"{row_b['model']} pred={row_b['prediction']} correct={row_b['correct']}"
    )
    text_h = 28
    draw.rectangle((0, 0, image.width, text_h), fill=(0, 0, 0))
    draw.text((8, 7), caption, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def classify(row_a: dict[str, str], row_b: dict[str, str], model_a: str, model_b: str) -> str:
    a_correct = row_a["correct"] == "1"
    b_correct = row_b["correct"] == "1"
    if a_correct and b_correct:
        return "both_correct"
    if not a_correct and not b_correct:
        return "both_wrong"
    if a_correct:
        return f"{model_a}_only_correct"
    return f"{model_b}_only_correct"


def sort_key(item: tuple[dict[str, str], dict[str, str], str], sort_by: str) -> tuple[float, str]:
    row_a, row_b, outcome = item
    if sort_by == "pair_id":
        return (0.0, row_a.get("pair_id", ""))
    wrong_row = row_b if outcome.endswith("_only_correct") and row_a["correct"] == "1" else row_a
    try:
        margin = float(wrong_row.get("relative_margin", "nan"))
    except ValueError:
        margin = math.nan
    if math.isnan(margin):
        margin = -1.0
    return (-margin, row_a.get("pair_id", ""))


def copy_optional(path: Path | None, dest_dir: Path) -> str:
    if path is None:
        return ""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if not dest.exists():
        shutil.copy2(path, dest)
    return dest.as_posix()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
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

    grouped: dict[str, list[tuple[dict[str, str], dict[str, str], str]]] = {}
    for key in common_keys:
        row_a = a_rows[key]
        row_b = b_rows[key]
        outcome = classify(row_a, row_b, args.model_a, args.model_b)
        grouped.setdefault(outcome, []).append((row_a, row_b, outcome))

    selected = []
    for outcome, items in grouped.items():
        if "only_correct" not in outcome and outcome != "both_wrong":
            continue
        items = sorted(items, key=lambda item: sort_key(item, args.sort_by))
        selected.extend(items[: args.limit_per_outcome])

    detail_rows = []
    annotated_dir = args.outdir / "annotated"
    depth_dir = args.outdir / "depth_pngs"
    for index, (row_a, row_b, outcome) in enumerate(selected, start=1):
        safe_pair_id = row_a.get("pair_id", f"{index:06d}").replace("/", "_")
        annotated_path = annotated_dir / f"{index:04d}_{outcome}_{safe_pair_id}.jpg"
        make_annotated_image(row_a, row_b, args.images_root, annotated_path)
        a_depth_png = copy_optional(depth_png_from_row(row_a), depth_dir / args.model_a)
        b_depth_png = copy_optional(depth_png_from_row(row_b), depth_dir / args.model_b)
        detail_rows.append(
            {
                "outcome": outcome,
                "image": row_a["image"],
                "pair_id": row_a.get("pair_id", ""),
                "closer": row_a["closer"],
                f"{args.model_a}_prediction": row_a["prediction"],
                f"{args.model_a}_correct": row_a["correct"],
                f"{args.model_a}_relative_margin": row_a["relative_margin"],
                f"{args.model_b}_prediction": row_b["prediction"],
                f"{args.model_b}_correct": row_b["correct"],
                f"{args.model_b}_relative_margin": row_b["relative_margin"],
                "annotated_image": annotated_path.as_posix(),
                f"{args.model_a}_depth_png": a_depth_png,
                f"{args.model_b}_depth_png": b_depth_png,
            }
        )

    fieldnames = [
        "outcome",
        "image",
        "pair_id",
        "closer",
        f"{args.model_a}_prediction",
        f"{args.model_a}_correct",
        f"{args.model_a}_relative_margin",
        f"{args.model_b}_prediction",
        f"{args.model_b}_correct",
        f"{args.model_b}_relative_margin",
        "annotated_image",
        f"{args.model_a}_depth_png",
        f"{args.model_b}_depth_png",
    ]
    write_csv(args.outdir / "disagreement_examples.csv", detail_rows, fieldnames)

    summary = {outcome: len(items) for outcome, items in grouped.items()}
    html_rows = []
    for row in detail_rows:
        annotated = html.escape(rel_to(Path(row["annotated_image"]), args.outdir))
        a_depth = html.escape(rel_to(Path(row[f"{args.model_a}_depth_png"]), args.outdir)) if row[f"{args.model_a}_depth_png"] else ""
        b_depth = html.escape(rel_to(Path(row[f"{args.model_b}_depth_png"]), args.outdir)) if row[f"{args.model_b}_depth_png"] else ""
        depth_cells = ""
        if a_depth:
            depth_cells += f'<div><b>{html.escape(args.model_a)}</b><br><img src="{a_depth}"></div>'
        if b_depth:
            depth_cells += f'<div><b>{html.escape(args.model_b)}</b><br><img src="{b_depth}"></div>'
        html_rows.append(
            f"""
            <section>
              <h2>{html.escape(row['outcome'])} / {html.escape(row['pair_id'])}</h2>
              <p>{html.escape(row['image'])}</p>
              <p>GT closer={html.escape(row['closer'])};
                 {html.escape(args.model_a)}={html.escape(row[f'{args.model_a}_prediction'])}
                 margin={html.escape(row[f'{args.model_a}_relative_margin'])};
                 {html.escape(args.model_b)}={html.escape(row[f'{args.model_b}_prediction'])}
                 margin={html.escape(row[f'{args.model_b}_relative_margin'])}</p>
              <div class="grid">
                <div><b>Original with DIW points</b><br><img src="{annotated}"></div>
                {depth_cells}
              </div>
            </section>
            """
        )

    summary_html = "".join(f"<li>{html.escape(k)}: {v}</li>" for k, v in sorted(summary.items()))
    report = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pair Disagreement Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    section {{ border-top: 1px solid #ddd; padding: 18px 0; }}
    img {{ max-width: 520px; max-height: 420px; object-fit: contain; border: 1px solid #ddd; }}
    .grid {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
  </style>
</head>
<body>
  <h1>Pair Disagreement Report</h1>
  <p>Models: {html.escape(args.model_a)} vs {html.escape(args.model_b)}</p>
  <ul>{summary_html}</ul>
  {''.join(html_rows)}
</body>
</html>
"""
    (args.outdir / "disagreement_report.html").write_text(report, encoding="utf-8")
    print(f"Wrote {args.outdir / 'disagreement_report.html'}")
    print(f"Wrote {args.outdir / 'disagreement_examples.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
