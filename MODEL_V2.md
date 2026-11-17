# BADI v2: description-based pattern complexity

**Status: proposed, uncalibrated model.** This changes the principal construct from total completion workload to the amount of pattern-specific information needed for exact reconstruction. The previous v1 workload is retained separately. No coefficient here was fitted to human performance; numerical tests are not psychometric validation.

## 1. Why v1 and an entropy/order score differ

V1 charges for every bead, every boundary and fragmentation, with a limited discount for short-period repetition. Even when its repetition factor P equals 1, 35% of its spatial penalty remains. A 100 x 100 checkerboard requires 10,000 placements, whereas the current seated design needs 4,753. Thus the previous total-score comparison confounded occupied count with information complexity.

A regular checkerboard has high boundary density and 1 bit/cell marginal binary color entropy, but knowing the alternation rule leaves no uncertainty in its interior. An independent equiprobable black/white grid has the same marginal entropy and still requires about 1 new bit/cell. Conditional entropy distinguishes these processes [1]. A finite exact pattern does not have a unique Shannon entropy without a probability model; v2 uses a finite-code description length as a reproducible proxy.

This is compatible with description-length approaches to regularity [2]. Research connecting simple descriptions to perceptual organization or concept learning motivates the idea, but does not validate this bead-scoring model [3,4].

## 2. Inputs and identities

Input is the final exact MARD code matrix, never a PDF screenshot, RGB illustration or compressed PNG file size. All 221 palette codes have fixed IDs 1..221 in the natural ordering used by v1. ID 0 denotes blank, not black. H7 is black and requires a bead.

Outer all-blank margins are cropped. Internal blank positions are part of the pattern and must be described. Let the resulting H x W rectangle have area A and N occupied cells. A silhouette-shaped checkerboard therefore retains the silhouette's descriptive cost, unlike a fully rectangular checkerboard.

For each of the eight rotations/reflections, labels are canonically renamed 0..K_total-1 in first-occurrence order. Here K_total includes blank when present; the reported color count excludes blank. The dictionary of actual palette IDs is encoded with a fixed number of bits per entry (8 for the MARD universe). The structural score depends on equality patterns, not code spelling or the RGB numerical value. A bijective recoloring that preserves the blank mask leaves structural complexity unchanged.

## 3. Finite minimum-description construction

For an exact grid G:

    L*(G) = min over tested orientation/model combinations
            [L(dimensions, dictionary, orientation, model) + L(payload)]

This is a computable restricted-model description length, not exact Kolmogorov complexity and not an optimal compressor over all possible programs. Adding an arbitrarily complex lookup table cannot make the pattern free: its dictionary, rule data, or residual must be transmitted. No pattern names, character identities or manual "portrait bonuses" enter the calculation.

### Header

Positive integers use Elias gamma coding: ell(n) = 2 floor(log2 n) + 1 bits.

For a palette universe of P colors, each actual color/blank ID uses b0 = ceil(log2(P+1)) bits. For MARD, P=221 and b0=8. The common header is:

    ell(H) + ell(W) + ell(K_total) + b0 K_total + 3 + 3

The last fields specify one of eight orientations and one of eight model tags (six currently used). Orientations and exact dictionaries are reversible. Dimensions refer to the tight unrotated rectangle; rotation determines the payload dimensions.

Let b = ceil(log2 K_total), with b=0 for one symbol.

### Candidate payloads

1. **Solid fill (tag 0).** Applicable when K_total=1. No payload beyond the dictionary and dimensions.
2. **Checkerboard (tag 1).** Applicable when the canonical grid equals (row+column) mod 2. No payload; dictionary order specifies the phase. This primitive avoids treating a complete alternation rule as thousands of separate changes.
3. **Small periodic tile plus exact exceptions (tag 2).** Test every tile dimension up to 8 x 8 (configurable 1..32). For each phase, select a modal observed symbol. Encode the tile height/width, all tile symbols, exception count, positive gaps between sorted raster exception positions, and exact replacement symbols. At a modal tie, canonical symbol order breaks the tie. All exceptions are paid for; one defective square does not destroy an otherwise regular model. The modal tile is a practical fit, not a claim of optimality among all residual codes.
4. **Run lengths (tag 3).** In the chosen raster orientation, encode total run count and each run's symbol and positive length. This supports large solid regions and stripes.
5. **Packed literal symbols (tag 4).** A*b bits. This is the fallback for uncompressible data; for K_total=1 it also uses zero payload.
6. **Raw DEFLATE of canonical one-byte symbols (tag 5).** Level 9, wbits=-15. Encode compressed byte count and exact compressed bytes. This detects additional repeated substrings and statistical redundancy not covered by the simple rules.

For the tile model, with dimensions p x q and e exceptions at increasing raster indices j_1..j_e (zero-based), define gaps d_1=j_1+1 and d_t=j_t-j_(t-1). Its payload length is:

    ell(p) + ell(q) + p*q*b + ell(e+1) + sum_t [ell(d_t) + b]

For the run model with R lengths n_r, payload length is:

    ell(R) + sum_r [b + ell(n_r)]

For DEFLATE of B bytes, payload length is ell(B+1)+8B.

The encoder explicitly writes the selected bitstream; the decoder verifies exact reconstruction and exact consumed bit count. Files are byte-padded only at the end. Reported `family_best_bits` are the best evaluated candidate lengths; candidates whose lower bound cannot beat the current winner are pruned, so these diagnostics are not claimed to be full family-wide minima. The chosen overall minimum is unaffected by this safe pruning.

### Scope and limitations of the grammar

The eight orientations are all tested, ensuring rotation/reflection invariance within this implementation. Canonical symbol mapping ensures code-renaming invariance. No lossy transformations or approximate recoloring are allowed in structural coding.

DEFLATE is an operational proxy for residual redundancy, not a simulation of a human. Some motifs, symmetries, hierarchies and semantic shapes will be easier for people than this grammar predicts. An expanded, validated grammar could include mirrored subregions or reusable local motifs. It must still pay to specify them. A seeded pseudo-random generator is deliberately NOT an allowed model: the random reference is reproducible for benchmarking, not assumed known to a maker as a free generative rule.

## 4. Normalization: pure structural score

Use the same MARD universe and fixed random reference as v1: independent uniform palette selections on 500 x 500 cells, generated as NumPy Generator(PCG64(1729)).integers(1,222,size=(500,500),dtype=uint32), then cast to int16.

Let L0 be the description of one H7 bead and Lref be the reference description. Under the supplied version:

- L0 = 17 bits.
- Lref = 1,961,922 bits.
- The reference decoder round trip is exact.

    u = max(0, (L*(G)-L0)/(Lref-L0))
    E = round_half_up[1 + 9999 * min(1,u)^gamma], gamma=0.5

All valid single-bead colors have the baseline E=1; the fixed reference is E=10,000. Above-reference ratios are retained even when the display saturates. Random samples differ, so the anchor is a benchmark, not a proven worst case. Implementations of DEFLATE can vary; software versions and the bit-length reference are recorded. Recalibrate/version the scale when changing the coding grammar, palette, software compressor behavior or reference.

Also report bits per active cell, L*/A, and occupied count N. The density includes the dimensional/dictionary header; it is not asserted to be an unbiased statistical entropy-rate estimate.

### What area means now

A random grid usually needs roughly A log2(P) pattern bits. A solid fill or perfect checkerboard needs primarily a rule, colors and dimensions, growing only logarithmically with size. Consequently, 500 x 500 regular patterns can have a low information-complexity score despite taking a long time to assemble. This is intentional. Strong area-linear manual-work penalties belong in the separate workload score, not in an entropy-like order score.

## 5. Keep color confusion separate, or explicitly label the combination

Recoloring an exact layout with similar shades does not alter its structural description. Nevertheless those shades may be harder to discriminate while making the craft. V2 retains the same Q as v1:

    S_ab = exp[-(DeltaE00(a,b)/6)^2], for different actual codes
    S_aa = 0; S_blank,a = 0

    q_edge(i) = mean four-neighbor S_(c_i,c_j)
    b_i = fraction of in-bounds neighbors with a different code
    q_tray(i) = (0.1+0.9*b_i) * max_other_used_code S_(c_i,other)
    Q = mean_occupied_i [0.75*q_edge(i)+0.25*q_tray(i)]

Color differences use the existing public sRGB palette converted to CIELAB D65/2-degree, then CIEDE2000 [5]. These are screen references, not measured physical bead differences. Q is a similarity penalty, not an error probability, and the width 6 is not a universal perception threshold.

Optionally define excess craft-adjusted information:

    J(G) = [L*(G)-L0] * (1 + lambda_Q*Q), lambda_Q=2 by default
    Jref = [Lref-L0] * (1 + lambda_Q*Qref)
    C = round_half_up[1+9999*min(1,max(0,J/Jref))^gamma]

This gives a higher craft-adjusted score to a near-color layout than to the same structure in distinguishable colors. The multiplier is bounded, so it does not turn a simple checkerboard into thousands of independent instructions. Unlike E, C is not purely a structural/entropy-like quantity. The user can set lambda_Q=0 to make it equal E.

**E and C use their own denominators. Compare patterns within a score column, not E against C as if C had to be numerically greater.** In particular the random 221-color reference itself has nonzero Q, affecting its calibration denominator.

## 6. Completion workload is an independent question

`bead_workload_v1.py` preserves v1. It remains a heuristic for placement-weighted effort, not elapsed time. Use `--include-workload` to report it alongside v2. Do not blend the displayed nonlinear scores by default; raw physical-work and descriptive-work terms can be calibrated against different human endpoints later.

Recommended output for a pattern:

    structural score E / craft-adjusted complexity C / beads N / separate workload

A checkerboard may be low E and substantial workload. A small detailed face may be high information density and relatively few placements. The seated design is more complex than both the full rectangular checkerboard and a checkerboard restricted to the same silhouette in the tests below.

## 7. Controlled comparisons and computed results

See `comparison_scores.csv` and individual `results/*.json`. Rounded defaults:

| Pattern | Beads | Colors | Description bits | Structural E | Craft-adjusted C | V1 workload |
|---|---:|---:|---:|---:|---:|---:|
| Single black | 1 | 1 | 17 | 1 | 1 | 1 |
| Solid black, 100 x 100 | 10,000 | 1 | 41 | 36 | 32 | 820 |
| Checkerboard, 100 x 100 | 10,000 | 2 | 51 | 43 | 38 | 1,170 |
| Same checkerboard, similar D2/H15 shades | 10,000 | 2 | 51 | 43 | 62 | 1,903 |
| Checkerboard, one error | 10,000 | 2 | 88 | 61 | 54 | 1,172 |
| Checkerboard, 5% randomly flipped cells | 10,000 | 2 | 4,230 | 464 | 413 | 1,545 |
| Checkerboard on seated design's exact mask | 4,753 | 2 | 2,090 | 326 | 290 | 813 |
| Regular 24-color tile on exact seated mask | 4,753 | 24 | 3,218 | 405 | 375 | 951 |
| Actual seated design | 4,753 | 24 | 13,974 | 844 | 774 | 960 |
| Original colors shuffled, same mask AND histogram | 4,753 | 24 | 23,168 | 1,087 | 1,015 | 1,219 |
| Amber-hat portrait | 9,500 | 22 | 12,030 | 783 | 711 | 1,105 |
| Zoomed moonlit portrait | 9,400 | 24 | 13,198 | 821 | 757 | 1,148 |
| Solid black, 500 x 500 | 250,000 | 1 | 49 | 41 | 37 | 4,095 |
| Checkerboard, 500 x 500 | 250,000 | 2 | 59 | 47 | 42 | 5,844 |
| Random 221-color reference | 250,000 | 221 | 1,961,922 | 10,000 | 10,000 | 10,000 |

A same-mask control preserves all 4,753 occupied positions and the empty silhouette background. It is not a fully filled board. The 24-color tile control preserves the code set but not its frequencies; the shuffled-original control preserves both frequencies and occupied mask. These test different confounders.

## 8. Computational verification, not human validation

The 25 checks include both anchors, exact decoder round trips, eight orientation invariances, blank-padding invariance, bijective palette-label invariance, unchanged inputs, preserved control counts, higher costs for selected exception patterns, and the relevant portrait/motif/shuffle inequalities. They verify these test cases, not a universal ranking of every artwork or monotonicity under every possible edit.

To validate with people, collect separate ratings of perceived order, pattern-memorization or reconstruction difficulty, completion time, and actual color/placement errors. Fix bead size, tools, labels, illumination and tray arrangement; include pattern-matched controls. Fit and cross-validate against held-out people and patterns. A user-preferred ordering criterion is a legitimate design requirement, but not independent evidence that a score is a universal measure of human difficulty.

No artistic or semantic quality is implied. The reference intentionally scores random noise as costly to reconstruct, even though noise has no meaningful subject. Do not optimize art by minimizing E or C alone; preserve image fidelity and protected features.

## 9. References and provenance

[1] Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal. Original paper reprint: https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf

[2] Grunwald, P. (2004). A tutorial introduction to the minimum description length principle. https://arxiv.org/abs/math/0406077

[3] Chater, N. (1996). Reconciling simplicity and likelihood principles in perceptual organization. Psychological Review, 103, 566–581. https://pubmed.ncbi.nlm.nih.gov/8759047/

[4] Feldman, J. (2000). Minimization of Boolean complexity in human concept learning. Nature, 407, 630–633. https://www.nature.com/articles/35036586 . These concept-learning experiments motivate the role of short descriptions; they did not study bead patterns.

[5] Sharma, G., Wu, W., and Dalal, E. N. (2005). The CIEDE2000 color-difference formula: implementation notes, supplementary test data, and mathematical observations. https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/

The palette and exact source grids are preserved from the original BADI v2 archive. Palette provenance, third-party notices, and file hashes are included in the repository. None of these references validates the proposed score, grammar or weights.
