#!/usr/bin/env python3
"""Build original README artwork and score its exact code matrices.

Run from the repository root: python docs/generate_gallery.py
The gallery artwork and constructed layouts are dedicated to CC0; see
docs/gallery/LICENSE.md. The existing palette and scoring code retain their
own terms.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bead_complexity import ComplexityModel, read_code_matrix  # noqa: E402

GALLERY = ROOT / "docs" / "gallery"
SIDE = 100
SHUFFLE_SEED = 20261117


def original_landscape() -> np.ndarray:
    """Draw a fictional mountain lake directly in exact MARD code IDs."""
    codes = ["C21", "C3", "C4", "A6", "H2", "C22", "C12", "C18",
             "B25", "C7", "C16", "B12", "B9"]
    color = {code: i for i, code in enumerate(codes)}
    canvas = Image.new("I", (SIDE, SIDE), color["C21"])
    pen = ImageDraw.Draw(canvas)
    pen.rectangle((0, 18, 99, 34), fill=color["C3"])
    pen.rectangle((0, 35, 99, 61), fill=color["C4"])
    pen.ellipse((69, 12, 85, 28), fill=color["A6"])
    for box in [(10, 19, 25, 23), (18, 16, 30, 23), (35, 10, 48, 14)]:
        pen.ellipse(box, fill=color["H2"])
    pen.polygon([(0, 55), (23, 27), (47, 58)], fill=color["C22"])
    pen.polygon([(16, 36), (23, 27), (31, 37), (24, 34), (21, 36)], fill=color["H2"])
    pen.polygon([(14, 67), (51, 29), (88, 67)], fill=color["C12"])
    pen.polygon([(51, 29), (54, 51), (88, 67)], fill=color["C18"])
    pen.polygon([(43, 38), (51, 29), (59, 38), (53, 36), (50, 39)], fill=color["H2"])
    pen.polygon([(60, 66), (86, 38), (99, 51), (99, 72)], fill=color["B25"])
    pen.rectangle((0, 65, 99, 99), fill=color["C7"])
    for y, left, right in [(70, 65, 86), (74, 69, 88), (78, 67, 82), (82, 73, 87)]:
        pen.line((left, y, right, y), fill=color["A6"])
    for y, left, right in [(68, 8, 29), (73, 34, 55), (80, 20, 42), (86, 44, 70)]:
        pen.line((left, y, right, y), fill=color["C16"])
    pen.polygon([(0, 83), (16, 79), (34, 89), (63, 92), (99, 86), (99, 99), (0, 99)], fill=color["B12"])
    for x, base, height in [(4, 87, 20), (14, 84, 28), (25, 91, 19), (91, 92, 21), (98, 91, 29)]:
        pen.rectangle((x, base - height // 2, x + 1, base + 3), fill=color["B9"])
        top = base - height
        for offset, half_width in [(0, 3), (height // 4, 5), (height // 2, 7)]:
            pen.polygon([(x, top + offset), (x - half_width, top + offset + height // 2),
                         (x + half_width, top + offset + height // 2)], fill=color["B9"])
    return np.asarray(codes)[np.asarray(canvas)]


def make_patterns() -> list[tuple[str, str, np.ndarray]]:
    yy, xx = np.indices((SIDE, SIDE))
    checker = np.where((xx + yy) % 2, "H7", "H2")
    flower = np.full((8, 8), "H2", dtype="<U3")
    flower[0:2, 2:4] = "A6"
    flower[2:4, 0:2] = "A6"
    flower[2:4, 4:6] = "A6"
    flower[4:6, 2:4] = "A6"
    flower[2:4, 2:4] = "C12"
    flower[6:8, 3] = "B8"
    flower[6, 4:6] = "B8"
    tiled = flower[yy % 8, xx % 8]
    landscape = original_landscape()
    rng = np.random.Generator(np.random.PCG64(SHUFFLE_SEED))
    shuffled = landscape.ravel()[rng.permutation(landscape.size)].reshape(landscape.shape)
    return [
        ("checkerboard", "Checkerboard", checker),
        ("flower-tile", "Repeating flower tile", tiled),
        ("mountain-lake", "Original mountain lake", landscape),
        ("shuffled-landscape", "Same landscape, shuffled", shuffled),
    ]


def render(grid: np.ndarray, palette: dict[str, str], destination: Path) -> None:
    # Display grid lines are cosmetic. Scores always use the exact CSV codes.
    colors = np.array([[tuple(bytes.fromhex(palette[code].removeprefix("#")))
                        for code in row] for row in grid], dtype=np.uint8)
    image = Image.fromarray(colors).resize((600, 600), Image.Resampling.NEAREST).convert("RGBA")
    lines = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(lines)
    for position in range(0, 600, 6):
        draw.line((position, 0, position, 599), fill=(255, 255, 255, 35))
        draw.line((0, position, 599, position), fill=(255, 255, 255, 35))
    image = Image.alpha_composite(image, lines).convert("RGB")
    image.save(destination, optimize=True)


def main() -> None:
    palette = json.loads((ROOT / "MARD_221_palette.json").read_text(encoding="utf-8"))
    model = ComplexityModel(palette)
    patterns = make_patterns()
    (GALLERY / "data").mkdir(parents=True, exist_ok=True)
    results = {}
    for name, title, grid in patterns:
        csv_path = GALLERY / "data" / f"{name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(grid.tolist())
        loaded = read_code_matrix(csv_path)
        assert np.array_equal(loaded, grid), f"CSV round trip failed: {name}"
        result = model.score(loaded)
        assert result["occupied_beads_N"] == SIDE * SIDE
        assert result["description"]["round_trip_exact"]
        render(grid, palette, GALLERY / f"{name}.png")
        results[name] = {"title": title, "grid_csv": f"data/{name}.csv", "image": f"{name}.png", **result}
        print(f"{title}: structural {result['structural_score']:,}; "
              f"{result['description']['description_bits']:,} bits; "
              f"{result['used_colors_K']} colors", flush=True)
    landscape, shuffled = patterns[2][2], patterns[3][2]
    assert np.array_equal(np.sort(landscape.ravel()), np.sort(shuffled.ravel()))
    scores = [results[name]["structural_score"] for name, _, _ in patterns]
    assert scores == sorted(scores), f"Gallery order changed: {scores}"
    manifest = {"artwork_license": "CC0-1.0", "side_cells": SIDE,
                "shuffle_seed": SHUFFLE_SEED, "patterns": results}
    (GALLERY / "scores.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Verified four exact CSV layouts, lossless codec round trips, equal bead counts, "
          "gallery order, and the landscape/shuffle color histogram.")


if __name__ == "__main__":
    main()
