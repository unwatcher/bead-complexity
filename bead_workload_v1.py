#!/usr/bin/env python3
"""BADI v1.0: transparent, UNCALIBRATED bead-assembly workload index.

Inputs are exact color-code matrices, never a screenshot or antialiased preview.
A blank cell is empty; H7/black is an occupied bead. See MODEL.md for equations.
Requires numpy, scipy and scikit-image. No network requests.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.measure import label

VERSION = 'BADI-1.0-provisional'
BLANK_TOKENS = {'', '.', '-', 'EMPTY', 'TRANSPARENT'}

@dataclass(frozen=True)
class Config:
    color_weight: float = 0.5
    entropy_weight: float = 1.5
    boundary_weight: float = 1.0
    fragmentation_weight: float = 1.5
    confusion_weight: float = 4.0
    palette_setup: float = 10.0
    blank_navigation: float = 0.05
    tau_delta_e: float = 6.0
    small_window: int = 5
    large_window: int = 11
    small_window_weight: float = 0.6
    repeat_discount: float = 0.65
    repeat_threshold: float = 0.9
    repeat_max_shift: int = 8
    boundary_confusion_share: float = 0.75
    minimum_palette_check: float = 0.1
    chart_confusion_factor: float = 1.0
    tray_confusion_factor: float = 1.0
    score_exponent: float = 0.5
    reference_side: int = 500
    reference_seed: int = 1729

    def validate(self) -> None:
        for key in ('color_weight','entropy_weight','boundary_weight',
                    'fragmentation_weight','confusion_weight','palette_setup',
                    'blank_navigation','chart_confusion_factor','tray_confusion_factor'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be nonnegative and finite')
        if (not math.isfinite(self.tau_delta_e) or not math.isfinite(self.score_exponent)
                or self.tau_delta_e <= 0 or self.score_exponent <= 0):
            raise ValueError('tau and score exponent must be positive')
        if self.reference_side < 2:
            raise ValueError('reference side must be >= 2')
        for value in (self.small_window, self.large_window):
            if value < 1 or value % 2 != 1:
                raise ValueError('entropy windows must be positive odd integers')
        for key in ('small_window_weight','repeat_discount','boundary_confusion_share',
                    'minimum_palette_check'):
            if not 0 <= getattr(self,key) <= 1:
                raise ValueError(f'{key} must be in [0,1]')
        if not 0 <= self.repeat_threshold < 1 or self.repeat_max_shift < 1:
            raise ValueError('invalid repeat settings')


def natural_key(code: str) -> tuple[str, int]:
    m = re.fullmatch(r'([A-Za-z]+)(\d+)', code)
    return (m[1].upper(), int(m[2])) if m else (code, 0)


def read_code_matrix(path: str | Path) -> np.ndarray:
    """Read raw matrix CSV or row/column-labeled CSV exported by our worksheets."""
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError('empty CSV')
    header = rows[0][0].strip().lower() if rows[0] else ''
    has_axes = header in {'row','row / column','row/column','r/c','row\\column'}
    if has_axes:
        wanted_cols = [str(i) for i in range(1, len(rows[0]))]
        if [x.strip() for x in rows[0][1:]] != wanted_cols:
            raise ValueError('column header must be consecutive 1-based coordinates')
        body = rows[1:]
        if any(not r or r[0].strip() != str(i+1) for i,r in enumerate(body)):
            raise ValueError('row header must be consecutive 1-based coordinates')
        rows = [r[1:] for r in body]
    if not rows or len(rows[0]) == 0 or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('matrix must be nonempty and rectangular')
    return np.array([[v.strip().upper() for v in r] for r in rows], dtype=str)


class DifficultyModel:
    def __init__(self, palette: dict[str, str], config: Config | None = None):
        self.config = config or Config()
        self.config.validate()
        self.palette = {str(k).upper(): str(v).upper() for k,v in palette.items()}
        if len(self.palette) < 2:
            raise ValueError('benchmark palette needs at least two colors')
        self.codes = sorted(self.palette, key=natural_key)
        for code in self.codes:
            if not re.fullmatch(r'#[0-9A-F]{6}', self.palette[code]):
                raise ValueError(f'invalid hex color for {code}')
        self.P = len(self.codes)
        self.code_to_id = {c:i+1 for i,c in enumerate(self.codes)}
        rgb = np.array([[int(self.palette[c][k:k+2], 16) for k in (1,3,5)]
                        for c in self.codes], dtype=float) / 255.
        labs = rgb2lab(rgb.reshape(1,-1,3), illuminant='D65', observer='2')[0]
        self.delta_e = np.zeros((self.P+1,self.P+1), dtype=float)
        self.delta_e[1:,1:] = deltaE_ciede2000(labs[:,None,:], labs[None,:,:])
        self.similarity = np.zeros_like(self.delta_e)
        self.similarity[1:,1:] = np.exp(-(self.delta_e[1:,1:]/self.config.tau_delta_e)**2)
        np.fill_diagonal(self.similarity, 0.)
        # Blank has no color and can NEVER produce a color-confusion penalty.
        self.similarity[0,:] = self.similarity[:,0] = 0.
        raw = json.dumps(self.palette, sort_keys=True, separators=(',',':')).encode()
        self.palette_sha256 = hashlib.sha256(raw).hexdigest()
        self._reference_result: dict[str, Any] | None = None

    def encode(self, matrix: np.ndarray) -> np.ndarray:
        a = np.asarray(matrix)
        if a.ndim != 2 or 0 in a.shape:
            raise ValueError('expected a nonempty two-dimensional code matrix')
        result = np.zeros(a.shape, dtype=np.int16)
        for code in np.unique(a.astype(str)):
            canon = str(code).strip().upper()
            if canon not in BLANK_TOKENS:
                if canon not in self.code_to_id:
                    raise ValueError(f'unknown code {canon!r}; no approximate remapping is allowed')
                result[a.astype(str) == code] = self.code_to_id[canon]
        if not np.any(result):
            raise ValueError('empty pattern has no score; the minimum is one bead')
        return result

    def benchmark_grid(self, seed: int | None = None) -> np.ndarray:
        cfg = self.config
        rng = np.random.Generator(np.random.PCG64(cfg.reference_seed if seed is None else seed))
        # uint32 calls specified for deterministic reproduction, then cast.
        return rng.integers(1, self.P+1, size=(cfg.reference_side,cfg.reference_side),
                            dtype=np.uint32).astype(np.int16)

    def _local_entropy(self, g: np.ndarray, side: int) -> np.ndarray:
        # In-bounds neighbors only. Internal blanks are a distinct categorical symbol.
        n = uniform_filter(np.ones(g.shape, float), size=side, mode='constant', cval=0)*side**2
        entropy = np.zeros(g.shape, float)
        for color in np.unique(g):
            count = uniform_filter((g == color).astype(float), size=side,
                                   mode='constant', cval=0)*side**2
            p = np.clip(count / n, 0., 1.)
            use = p > 1e-12
            entropy[use] -= p[use]*np.log2(p[use])
        cap = np.minimum(n, self.P + int(np.any(g == 0)))
        den = np.log2(np.maximum(cap,1.))
        return np.clip(np.divide(entropy,den,out=np.zeros_like(entropy),where=den>0),0.,1.)

    @staticmethod
    def _overlap(g: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray,np.ndarray]:
        h,w = g.shape
        if abs(dy) >= h or abs(dx) >= w:
            return g[:0,:0],g[:0,:0]
        ya,yb = max(0,-dy), min(h,h-dy)
        xa,xb = max(0,-dx), min(w,w-dx)
        return g[ya:yb,xa:xb],g[ya+dy:yb+dy,xa+dx:xb+dx]

    def _repeatability(self, g: np.ndarray, N: int) -> tuple[float,dict[str,Any]]:
        cfg = self.config
        best, detail = 0., {}
        for k in range(1,cfg.repeat_max_shift+1):
            for dy,dx in ((0,k),(k,0),(k,k),(k,-k)):
                a,b = self._overlap(g,dy,dx)
                valid = (a > 0)&(b > 0)
                n = int(valid.sum())
                if n < max(16, math.ceil(N/2)):
                    continue
                x,y = a[valid],b[valid]
                pa = np.bincount(x,minlength=self.P+1)/n
                pb = np.bincount(y,minlength=self.P+1)/n
                chance = float(pa@pb)
                if chance >= 1.-1e-12:
                    continue  # uniform fills need no bonus: their structural cost is already zero
                match = float(np.mean(x == y))
                z = (match-chance)/(1.-chance)
                if z > best:
                    best = z
                    detail = {'shift_dy_dx':[dy,dx],'paired_cells':n,'match_fraction':match,
                              'chance_match':chance,'corrected_match':z}
        bonus = float(np.clip((best-cfg.repeat_threshold)/(1.-cfg.repeat_threshold),0.,1.))
        return bonus,detail

    def evaluate_encoded(self, input_grid: np.ndarray, *, keep_maps: bool = False) -> dict[str,Any]:
        cfg = self.config
        inp = np.asarray(input_grid)
        if inp.ndim != 2 or 0 in inp.shape or not np.issubdtype(inp.dtype,np.integer):
            raise ValueError('encoded grid must be a nonempty 2D integer array')
        if inp.min() < 0 or inp.max() > self.P or not np.any(inp):
            raise ValueError('invalid palette IDs or empty pattern')
        ys,xs = np.nonzero(inp)
        y0,y1,x0,x1 = ys.min(),ys.max()+1,xs.min(),xs.max()+1
        g = inp[y0:y1,x0:x1].copy()
        occupied = g > 0
        N,A = int(occupied.sum()),int(g.size)
        counts = np.bincount(g[occupied],minlength=self.P+1)
        used = np.flatnonzero(counts)
        K = len(used)
        p = counts[used]/N
        global_entropy = float(-(p*np.log2(p)).sum())
        color_load = (math.log2(K)+global_entropy)/(2*math.log2(self.P))
        local_h = (cfg.small_window_weight*self._local_entropy(g,cfg.small_window)+
                   (1-cfg.small_window_weight)*self._local_entropy(g,cfg.large_window))

        degree = np.zeros(g.shape,float)
        boundary = np.zeros(g.shape,float)
        close_edges = np.zeros(g.shape,float)
        pair_count = np.zeros((self.P+1,self.P+1),dtype=np.int64)
        # Undirected in-bounds horizontal and vertical edges, then credited to both ends.
        for axis in (0,1):
            sl1 = (slice(None,-1),slice(None)) if axis == 0 else (slice(None),slice(None,-1))
            sl2 = (slice(1,None),slice(None)) if axis == 0 else (slice(None),slice(1,None))
            a,b = g[sl1],g[sl2]
            diff = (a != b).astype(float)
            sim = self.similarity[a,b]
            degree[sl1] += 1; degree[sl2] += 1
            boundary[sl1] += diff; boundary[sl2] += diff
            close_edges[sl1] += sim; close_edges[sl2] += sim
            ok = (a > 0)&(b > 0)&(a != b)
            x,y = np.minimum(a[ok],b[ok]),np.maximum(a[ok],b[ok])
            pair_count += np.bincount(x.astype(np.int64)*(self.P+1)+y,minlength=(self.P+1)**2).reshape(self.P+1,self.P+1)
        boundary = np.divide(boundary,degree,out=np.zeros_like(boundary),where=degree>0)
        close_edges = np.divide(close_edges,degree,out=np.zeros_like(close_edges),where=degree>0)

        cc = label(g,background=0,connectivity=1)
        sizes = np.bincount(cc.ravel())
        region_count = len(sizes)-1
        fragmentation = np.zeros(g.shape,float)
        if N > 1:
            fragmentation[occupied] = ((1/np.sqrt(sizes[cc[occupied]])-1/math.sqrt(N))/
                                      (1-1/math.sqrt(N)))
        hardest_competitor = self.similarity[:,used].max(axis=1)
        check_rate = cfg.minimum_palette_check+(1-cfg.minimum_palette_check)*boundary
        tray_confusion = check_rate*hardest_competitor[g]
        q = (cfg.boundary_confusion_share*cfg.chart_confusion_factor*close_edges+
             (1-cfg.boundary_confusion_share)*cfg.tray_confusion_factor*tray_confusion)
        repeat,repeat_detail = self._repeatability(g,N)
        structural_factor = 1-cfg.repeat_discount*repeat
        per_bead = (1+cfg.color_weight*color_load+
                    structural_factor*(cfg.entropy_weight*local_h+cfg.boundary_weight*boundary+
                                       cfg.fragmentation_weight*fragmentation)+
                    cfg.confusion_weight*q)
        mean = lambda x: float(np.mean(x[occupied]))
        components = {
            'base_placement':float(N),
            'palette_management':float(N*cfg.color_weight*color_load),
            'local_entropy':float(N*structural_factor*cfg.entropy_weight*mean(local_h)),
            'boundaries':float(N*structural_factor*cfg.boundary_weight*mean(boundary)),
            'fragmentation':float(N*structural_factor*cfg.fragmentation_weight*mean(fragmentation)),
            'similar_color_confusion':float(N*cfg.confusion_weight*mean(q)),
            'palette_setup':float(cfg.palette_setup*(K-1)),
            'blank_navigation':float(cfg.blank_navigation*(A-N)),
        }
        raw = sum(components.values())
        nearest_pairs = []
        for ai,a in enumerate(used):
            for b in used[ai+1:]:
                e = int(pair_count[a,b])
                nearest_pairs.append({'code_a':self.codes[a-1],'code_b':self.codes[b-1],
                                     'delta_e_00':float(self.delta_e[a,b]),
                                     'confusability_kernel':float(self.similarity[a,b]),
                                     'touching_edges':e,'edge_risk_mass':float(e*self.similarity[a,b])})
        result = {
            'model_version':VERSION,'grid_shape_H_W':list(map(int,inp.shape)),
            'active_bbox_xyxy_zero_based':[int(x0),int(y0),int(x1),int(y1)],
            'active_grid_shape_H_W':list(g.shape),'bounding_area_A':A,'occupied_beads_N':N,
            'empty_cells_in_active_box':A-N,'used_colors_K':K,
            'effective_colors':float(2**global_entropy),'global_color_entropy_bits':global_entropy,
            'color_load_C':float(color_load),'local_entropy_L':mean(local_h),
            'boundary_density_B':mean(boundary),'fragmentation_F':mean(fragmentation),
            'monochrome_components':region_count,
            'single_bead_components':int(np.sum(sizes[1:] == 1)),
            'repeat_bonus_P':repeat,'repeat_detail':repeat_detail,
            'local_confusion_Qedge':mean(close_edges),'palette_confusion_Qtray':mean(tray_confusion),
            'combined_confusion_Q':mean(q),'raw_workload_W':raw,
            'workload_per_bead':raw/N,'local_intensity_mean':mean(per_bead),
            'local_intensity_95th_percentile':float(np.percentile(per_bead[occupied],95)),
            'workload_components':components,
            'closest_used_color_pairs':sorted(nearest_pairs,key=lambda z:z['delta_e_00'])[:10],
            'highest_spatial_confusion_pairs':sorted(nearest_pairs,key=lambda z:-z['edge_risk_mass'])[:10],
            'palette_sha256':self.palette_sha256,'config':asdict(cfg),
            'encoded_grid_sha256':hashlib.sha256(g.astype('<i2').tobytes()).hexdigest(),
        }
        if keep_maps:
            result['_maps'] = {'per_bead_workload':np.where(occupied,per_bead,np.nan),
                               'entropy':local_h,'boundaries':boundary,'fragmentation':fragmentation,
                               'confusion':q,'encoded_grid':g}
        return result

    def reference(self) -> dict[str,Any]:
        if self._reference_result is None:
            self._reference_result = self.evaluate_encoded(self.benchmark_grid())
        return self._reference_result

    def score(self, matrix: np.ndarray, *, encoded: bool = False, keep_maps: bool = False) -> dict[str,Any]:
        g = matrix if encoded else self.encode(matrix)
        result = self.evaluate_encoded(g,keep_maps=keep_maps)
        reference = self.reference()
        W,Wr = result['raw_workload_W'],reference['raw_workload_W']
        relative = max(0.,(W-1)/(Wr-1))
        raw_score = 1+9999*relative**self.config.score_exponent
        display_score = 1+9999*min(relative,1.)**self.config.score_exponent
        result.update({'difficulty_score':int(math.floor(display_score+0.5)),
                       'unclipped_score':raw_score,'above_reference':relative>1.000000001,
                       'reference_workload':Wr,'reference_grid_sha256':reference['encoded_grid_sha256']})
        return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid',type=Path,required=True,help='exact code matrix CSV')
    p.add_argument('--palette',type=Path,default=Path(__file__).with_name('MARD_221_palette.json'))
    p.add_argument('--config',type=Path,help='JSON object overriding Config fields')
    p.add_argument('--out',type=Path,default=Path('difficulty_results.json'))
    p.add_argument('--cell-map',type=Path,help='optional CSV of local workload values')
    args = p.parse_args()
    cfg = Config(**json.loads(args.config.read_text())) if args.config else Config()
    model = DifficultyModel(json.loads(args.palette.read_text()),cfg)
    result = model.score(read_code_matrix(args.grid),keep_maps=bool(args.cell_map))
    if args.cell_map:
        maps = result.pop('_maps')
        args.cell_map.parent.mkdir(parents=True,exist_ok=True)
        np.savetxt(args.cell_map,maps['per_bead_workload'],delimiter=',',fmt='%.6f')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(f"BADI {result['difficulty_score']:,}/10,000 | "
          f"{result['occupied_beads_N']:,} beads | {result['used_colors_K']} colors | "
          f"{result['workload_per_bead']:.3f} work units/bead (uncalibrated)")
    print(args.out)

if __name__ == '__main__':
    main()
