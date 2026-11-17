# Visual complexity examples

The four PNGs are original, procedurally constructed bead layouts dedicated
to [CC0](LICENSE.md). Each depicts an exact 100 × 100 matrix containing
10,000 occupied bead positions. White backgrounds also count as beads.
The thin display grid is a cosmetic overlay.

`data/*.csv` holds the exact color-code matrices. `scores.json` records the
measured BADI scores, bit counts, runtime versions, palette/reference hashes,
and decoder round-trip checks. The landscape and shuffled landscape preserve
the same color histogram; their structural arrangement is the only change.

To recreate the images, CSVs, and scores from the repository root:

```sh
python -m pip install -r requirements.txt
python -m pip install Pillow
python docs/generate_gallery.py
```

The captions in the main README show approximate structural scores from one
recorded run. Different runtime/compressor versions may move the values
slightly. These examples illustrate the model rather than a universal human
difficulty scale. The 10,000 anchor is a much larger 500 × 500 random pattern;
the gallery is not intended to fill the entire score range.
