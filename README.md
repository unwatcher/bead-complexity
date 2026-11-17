# Bead Complexity · BADI v2

**Order-aware complexity scoring for Perler and fuse-bead patterns.**

BADI v2 measures the information needed to reconstruct an exact bead layout. A large checkerboard can be described by a short rule; an irregular portrait needs many more pattern-specific instructions. The algorithm captures that difference while reporting color confusion and physical placement workload separately.

Use it to compare exact patterns, study repetition and irregularity, or add explainable complexity metrics to a bead-pattern application. Evaluation runs locally, with no API keys or network service required.

> **Experimental model.** Scores are reproducible, but have not been calibrated against human ratings, assembly time, or error rates.

## What the algorithm reports

| Output | Meaning |
| --- | --- |
| `structural_score` | A 1–10,000 score based on the shortest tested lossless description. Regularity reduces the information needed to specify a pattern. |
| `craft_adjusted_complexity_score` | A separately normalized score that also accounts for ambiguity between similar palette colors. |
| `occupied_beads_N` | The number of physical bead placements. Empty cells are excluded. |
| `description` | Bit count, selected encoding, orientation, information density, and decoder round-trip status. |
| `workload_v1` | Optional placement-weighted effort metrics from the earlier BADI model. |

The two complexity scores use different normalization denominators. Compare patterns within the same score column; subtracting one score from the other does not measure a difficulty increment. Both use a fixed, seeded 500 × 500 random reference as the 10,000 anchor, rather than a theoretical maximum.

The encoder tests solid fills, checkerboards, periodic tiles with exact exceptions, run lengths, packed symbols, and DEFLATE across eight rotations/reflections. Each candidate pays for its dimensions, palette dictionary, rules, and payload. The selected binary description is decoded and checked against the exact input layout.

## Example: same bead count, different structure

These bundled examples share the same 4,753 occupied positions. The portrait and shuffled portrait also have identical color counts and frequencies.

| Pattern | Beads | Colors | Description bits | Structural score |
| --- | ---: | ---: | ---: | ---: |
| Checkerboard within the portrait silhouette | 4,753 | 2 | 2,090 | 326 |
| Repeating 24-color tile within the silhouette | 4,753 | 24 | 3,218 | 405 |
| Original portrait | 4,753 | 24 | 13,974 | 844 |
| Shuffled portrait, preserving silhouette and color histogram | 4,753 | 24 | 23,168 | 1,087 |

The tile control preserves the color set, while the shuffled control preserves the full color histogram. See [all comparisons](comparison_scores.csv) and the [model specification](MODEL_V2.md) for assumptions and equations. Recorded values can vary with compressor and library versions; each result includes runtime metadata.

## Quick start

Requires Python 3.10 or newer and NumPy, SciPy, and scikit-image.

```sh
git clone https://github.com/unwatcher/bead-complexity.git
cd bead-complexity
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then install dependencies and score a sample:

```sh
python -m pip install -r requirements.txt
python bead_complexity.py --grid example_inputs/seated_girl_latest.csv --out generated/seated.json
```

Include placement workload and save a reversible binary description:

```sh
python bead_complexity.py --grid example_inputs/seated_girl_latest.csv --include-workload --description-out generated/seated.bpc --out generated/seated.json
```

Use `--config config_v2.json` to specify the model settings and `--palette MARD_221_palette.json` to select the supplied palette explicitly. The first score for a model instance also computes its reference pattern.

## Input format

Input is a rectangular CSV matrix of exact palette codes. Both plain matrices and CSVs with consecutive one-based row/column labels are supported:

```csv
row,1,2,3
1,H7,H2,
2,H2,H7,H2
```

- The bundled palette contains 221 MARD codes. Unknown codes raise an error; they are not approximately remapped.
- Blank cells, `.`, `-`, `EMPTY`, and `TRANSPARENT` represent empty positions. `H7` is black and counts as a bead.
- Outer empty margins are cropped. Internal empty cells and the silhouette remain part of the pattern.
- Supply the final code matrix. A screenshot, photograph, or rendered worksheet must first be converted into exact palette codes outside this tool.

## Python API

```python
import json
from pathlib import Path
from bead_complexity import ComplexityModel, read_code_matrix

palette = json.loads(Path("MARD_221_palette.json").read_text())
model = ComplexityModel(palette)
grid = read_code_matrix("example_inputs/seated_girl_latest.csv")
result = model.score(grid, include_v1_workload=True)

print(result["structural_score"])
print(result["occupied_beads_N"])
```

Decode a saved description using the identical palette definition:

```python
grid_ids, bits_consumed = model.decode(Path("generated/seated.bpc").read_bytes())
# ID 0 is empty; IDs 1..221 follow model.base.codes.
```

A `.bpc` file stores the tightly cropped code layout, including its original orientation. It does not preserve unused outer margins and is not an image or worksheet. Scoring uses consumed bits rather than the byte-padded file size.

## Reproduce the examples

```sh
python run_examples.py
```

This regenerates the comparison table, JSON results, binary descriptions, matched controls, benchmark, and `tests.json`. The 25 checks cover normalization anchors, exact decoder round trips, all eight orientation invariances, empty-padding and palette-label invariance, preserved inputs, and selected regularity/shuffle comparisons. They test these computational properties and examples, not a universal ranking of artwork or human performance.

Changing the palette, grammar, reference, or scoring parameters creates a different scoring convention. Record those changes before comparing results across versions.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| [bead_complexity.py](bead_complexity.py) | BADI v2 evaluator, lossless encoder/decoder, and CLI |
| [bead_workload_v1.py](bead_workload_v1.py) | Earlier workload model, retained for separate comparisons |
| [MODEL_V2.md](MODEL_V2.md) | Mathematical definitions, coding grammar, limitations, and research references |
| [config_v2.json](config_v2.json) | Default settings |
| [example_inputs/](example_inputs/) | Exact source patterns and controlled comparisons |
| [results/](results/) | Recorded scores and binary descriptions |
| [benchmark.json](benchmark.json), [tests.json](tests.json) | Reference values and recorded computational checks |
| [SHA256.json](SHA256.json) | SHA-256 file manifest for this repository snapshot |

## Scope and attribution

Description length is a restricted-model proxy for structural complexity, not exact Kolmogorov complexity or a direct measure of artistic quality. Semantic meaning, maker experience, tools, lighting, and physical bead variation are outside this model. Palette RGB values are screen references rather than measurements of physical beads.

The bundled palette is derived from [pixel-to-beads](https://github.com/a31521424/pixel-to-beads). Its source, transformations, and GPL-3.0 license copy are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No project-wide license has been selected for the original algorithm source.
