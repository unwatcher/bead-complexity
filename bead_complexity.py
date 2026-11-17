#!/usr/bin/env python3
"""Order-aware bead-pattern complexity, v2.0 (PROVISIONAL / not human-validated).

Scores exact label matrices with a small lossless model bank rather than counting
all transitions as independent mental decisions. Also reports the v1-compatible
near-color ambiguity Q. The separate v1 workload model is retained, not replaced
by an assertion that complexity bits predict time. See MODEL_V2.md.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np
from bead_workload_v1 import DifficultyModel, read_code_matrix

VERSION = 'BADI-2.0-order-aware-provisional'

@dataclass(frozen=True)
class Config:
    max_tile_side: int = 8
    confusion_weight: float = 2.0
    score_exponent: float = 0.5
    reference_side: int = 500
    reference_seed: int = 1729

    def validate(self):
        if not 1 <= self.max_tile_side <= 32:
            raise ValueError('max_tile_side must be 1..32')
        if self.reference_side < 2:
            raise ValueError('reference_side must be >= 2')
        if not math.isfinite(self.confusion_weight) or self.confusion_weight < 0:
            raise ValueError('confusion_weight must be nonnegative')
        if not math.isfinite(self.score_exponent) or self.score_exponent <= 0:
            raise ValueError('score_exponent must be positive')


def gamma_length(n: int) -> int:
    if n < 1: raise ValueError('gamma code requires positive integers')
    return 2 * int(n).bit_length() - 1


def gamma_sum(x: np.ndarray) -> int:
    if not x.size: return 0
    return int(np.sum(2 * np.floor(np.log2(x)).astype(np.int64) + 1))


def transform(g, t):
    a = np.rot90(g, t % 4)
    return np.fliplr(a) if t >= 4 else a


def inverse_transform(g, t):
    return np.rot90(np.fliplr(g) if t >= 4 else g, -(t % 4))


def canonical_labels(g):
    """First-appearance labels: independent of spelling, ID magnitude and RGB."""
    values, first, inverse = np.unique(g, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank[inverse].reshape(g.shape).astype(np.uint8), values[order]


class Bits:
    def __init__(self): self.parts = []
    def fixed(self, n, width):
        n = int(n)
        if width == 0:
            if n: raise ValueError('nonzero value in zero bits')
            return
        if not 0 <= n < (1 << width): raise ValueError('out of range')
        self.parts.append(format(n, f'0{width}b'))
    def gamma(self, n):
        n = int(n)
        if n < 1: raise ValueError('gamma needs positive integer')
        b = bin(n)[2:]
        self.parts.append('0' * (len(b)-1) + b)
    def binary(self, payload):
        self.parts.extend(format(v, '08b') for v in payload)
    def finish(self):
        s = ''.join(self.parts); n = len(s)
        padded = s + '0' * ((-n) % 8)
        return bytes(int(padded[i:i+8],2) for i in range(0,len(padded),8)), n


class Reader:
    def __init__(self, payload):
        self.s = ''.join(format(v,'08b') for v in payload); self.i = 0
    def fixed(self, width):
        if width == 0: return 0
        if self.i + width > len(self.s): raise ValueError('truncated description')
        n = int(self.s[self.i:self.i+width],2); self.i += width
        return n
    def gamma(self):
        zeros = 0
        while self.fixed(1) == 0:
            zeros += 1
            if zeros > 40: raise ValueError('invalid gamma value')
        return (1 << zeros) + self.fixed(zeros)
    def binary(self, n): return bytes(self.fixed(8) for _ in range(n))


class ComplexityModel:
    def __init__(self, palette, config=None):
        self.config = config or Config(); self.config.validate()
        self.base = DifficultyModel(palette)
        if self.base.P > 255:
            raise ValueError('this byte-label codec supports at most 255 colors plus blank')
        self.symbol_id_bits = math.ceil(math.log2(self.base.P+1))
        self._reference = None
        self.single_bits = self.describe(np.array([[self.base.code_to_id['H7']]], dtype=np.int16))['description_bits']

    @staticmethod
    def tight_crop(g):
        g = np.asarray(g)
        if g.ndim != 2 or not g.size or not np.any(g):
            raise ValueError('a nonempty 2D pattern with at least one bead is required')
        yy,xx = np.nonzero(g)
        box = [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        return g[box[1]:box[3],box[0]:box[2]].astype(np.int16), box

    def describe(self, encoded_grid):
        """Select shortest tested lossless encoding; header and rules are paid for."""
        g, box = self.tight_crop(encoded_grid)
        H,W = g.shape; K = int(len(np.unique(g))); b = (K-1).bit_length()
        # gamma H, W, K; K actual palette/blank IDs; 3 orientation + 3 model bits.
        common = gamma_length(H)+gamma_length(W)+gamma_length(K)+K*self.symbol_id_bits+6
        best = {'description_bits': common + g.size*b, 'model':'raw', 'orientation':0}
        best_spec = None
        family_best = {}
        for t in range(8):
            a, dictionary = canonical_labels(transform(g,t))
            h,w = a.shape; flat = a.ravel(); A = a.size
            specs = [{'model':'raw','tag':4,'payload_bits':A*b}]
            if K == 1: specs.insert(0,{'model':'solid','tag':0,'payload_bits':0})
            yy,xx = np.indices(a.shape)
            if K == 2 and np.array_equal(a, (yy+xx)%2):
                specs.append({'model':'checkerboard','tag':1,'payload_bits':0})
            changes = np.flatnonzero(flat[1:] != flat[:-1])+1
            starts = np.r_[0,changes]; lengths = np.diff(np.r_[starts,A]); values=flat[starts]
            specs.append({'model':'run_length','tag':3,
                          'payload_bits':gamma_length(len(starts)) + len(starts)*b + gamma_sum(lengths),
                          'lengths':lengths,'values':values})
            compressor=zlib.compressobj(level=9, wbits=-15)
            payload=compressor.compress(flat.tobytes())+compressor.flush()
            specs.append({'model':'deflate','tag':5,
                          'payload_bits':gamma_length(len(payload)+1)+8*len(payload), 'payload':payload})
            for ph in range(1,min(h,self.config.max_tile_side)+1):
                for pw in range(1,min(w,self.config.max_tile_side)+1):
                    overhead=gamma_length(ph)+gamma_length(pw)+ph*pw*b
                    if common+overhead+1 >= best['description_bits']:
                        continue
                    phase=((yy%ph)*pw+(xx%pw)).ravel()
                    counts=np.bincount(phase*K+flat, minlength=ph*pw*K).reshape(ph*pw,K)
                    tile=counts.argmax(axis=1).astype(np.uint8)
                    pos=np.flatnonzero(flat != tile[phase]); e=len(pos)
                    # Each exception gap is >=1 bit; each replacement is b bits.
                    lower=overhead+gamma_length(e+1)+e*(1+b)
                    if common+lower >= best['description_bits']:
                        continue
                    gaps=np.diff(np.r_[-1,pos])
                    payload_bits=overhead+gamma_length(e+1)+gamma_sum(gaps)+e*b
                    specs.append({'model':'tile_exceptions','tag':2,'payload_bits':payload_bits,
                                  'ph':ph,'pw':pw,'tile':tile,'positions':pos,'gaps':gaps,
                                  'values':flat[pos]})
            for s in specs:
                length=int(common+s['payload_bits'])
                family_best[s['model']]=min(family_best.get(s['model'],length),length)
                if best_spec is None or length < best['description_bits']:
                    best={'description_bits':length,'model':s['model'],'orientation':t}
                    best_spec=(s,a.copy(),dictionary.copy())
        assert best_spec is not None
        s,a,dictionary=best_spec
        bw=Bits(); bw.gamma(H); bw.gamma(W); bw.gamma(K)
        for code in dictionary: bw.fixed(code,self.symbol_id_bits)
        bw.fixed(best['orientation'],3); bw.fixed(s['tag'],3)
        if s['model']=='raw':
            for v in a.ravel(): bw.fixed(v,b)
        elif s['model']=='run_length':
            bw.gamma(len(s['values']))
            for v,n in zip(s['values'],s['lengths']): bw.fixed(v,b); bw.gamma(n)
        elif s['model']=='deflate':
            bw.gamma(len(s['payload'])+1); bw.binary(s['payload'])
        elif s['model']=='tile_exceptions':
            bw.gamma(s['ph']); bw.gamma(s['pw'])
            for v in s['tile']: bw.fixed(v,b)
            bw.gamma(len(s['positions'])+1)
            for gap,v in zip(s['gaps'],s['values']): bw.gamma(gap); bw.fixed(v,b)
        blob,n= bw.finish()
        if n != best['description_bits']: raise AssertionError('bit-budget mismatch')
        decoded, consumed = self.decode(blob)
        if consumed != n or not np.array_equal(decoded,g): raise AssertionError('lossless round-trip failed')
        info={**best,'header_bits':int(common),'payload_bits':int(n-common),
              'active_shape_H_W':list(map(int,g.shape)),'active_bbox_xyxy':box,
              'alphabet_size_including_blank':K,'bits_per_active_cell':n/g.size,
              'round_trip_exact':True,'consumed_bits':n,'stored_bytes_including_padding':len(blob),
              'family_best_bits':family_best,'_description_bytes':blob}
        if s['model']=='tile_exceptions':
            info.update({'tile_shape_H_W':[s['ph'],s['pw']], 'exception_cells':len(s['positions'])})
        return info

    def decode(self, payload):
        r=Reader(payload); H,W,K=r.gamma(),r.gamma(),r.gamma()
        if H*W > 1_000_000 or K > self.base.P+1:
            raise ValueError('description dimensions/alphabet exceed safe decoder limits')
        dictionary=np.array([r.fixed(self.symbol_id_bits) for _ in range(K)],dtype=np.int16)
        t,tag=r.fixed(3),r.fixed(3); h,w=(W,H) if t%2 else (H,W); A=h*w; b=(K-1).bit_length()
        if tag==0: a=np.zeros((h,w),np.uint8)
        elif tag==1:
            yy,xx=np.indices((h,w)); a=((yy+xx)%2).astype(np.uint8)
        elif tag==2:
            ph,pw=r.gamma(),r.gamma()
            if ph>h or pw>w: raise ValueError('invalid tile')
            tile=np.array([r.fixed(b) for _ in range(ph*pw)],dtype=np.uint8).reshape(ph,pw)
            yy,xx=np.indices((h,w)); a=tile[yy%ph,xx%pw]; flat=a.ravel(); e=r.gamma()-1; pos=-1
            if e>A: raise ValueError('too many exceptions')
            for _ in range(e):
                pos+=r.gamma()
                if pos>=A: raise ValueError('exception position out of bounds')
                flat[pos]=r.fixed(b)
        elif tag==3:
            nrun=r.gamma(); values=[]; total=0
            if nrun>A: raise ValueError('too many runs')
            for _ in range(nrun):
                v,n=r.fixed(b),r.gamma(); total+=n
                if total>A: raise ValueError('run overflow')
                values.extend([v]*n)
            if total!=A: raise ValueError('run underflow')
            a=np.array(values,dtype=np.uint8).reshape(h,w)
        elif tag==4: a=np.array([r.fixed(b) for _ in range(A)],dtype=np.uint8).reshape(h,w)
        elif tag==5:
            size=r.gamma()-1
            if size>A*3+1000: raise ValueError('compressed payload too large')
            obj=zlib.decompressobj(wbits=-15); raw=obj.decompress(r.binary(size),A+1)
            if len(raw)!=A or not obj.eof or obj.unused_data: raise ValueError('invalid deflate stream')
            a=np.frombuffer(raw,np.uint8).reshape(h,w)
        else: raise ValueError('unknown model tag')
        if a.max()>=K: raise ValueError('invalid symbol')
        return inverse_transform(dictionary[a],t),r.i

    def ambiguity(self, g):
        occ=g>0; used=np.unique(g[occ]); degree=np.zeros(g.shape); boundary=np.zeros(g.shape); edge=np.zeros(g.shape)
        for axis in (0,1):
            s1=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
            s2=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
            a,b=g[s1],g[s2]; d=(a!=b); close=self.base.similarity[a,b]
            for s in (s1,s2): degree[s]+=1; boundary[s]+=d; edge[s]+=close
        boundary=np.divide(boundary,degree,out=np.zeros_like(boundary),where=degree>0)
        edge=np.divide(edge,degree,out=np.zeros_like(edge),where=degree>0)
        alternatives=self.base.similarity[:,used].max(axis=1)
        tray=(0.1+0.9*boundary)*alternatives[g]
        Q=float(np.mean((0.75*edge+0.25*tray)[occ]))
        return Q, float(np.mean(boundary[occ]))

    def reference(self):
        if self._reference is None:
            rng=np.random.Generator(np.random.PCG64(self.config.reference_seed))
            g=rng.integers(1,self.base.P+1,size=(self.config.reference_side,)*2,dtype=np.uint32).astype(np.int16)
            d=self.describe(g); d.pop('_description_bytes')
            q,b=self.ambiguity(g)
            self._reference={'description':d,'Q':q,'grid_sha256':hashlib.sha256(g.astype('<i2').tobytes()).hexdigest()}
        return self._reference

    def score(self, matrix, *, encoded=False, keep_description=False, include_v1_workload=False):
        g=np.asarray(matrix) if encoded else self.base.encode(matrix)
        if g.ndim!=2 or not np.issubdtype(g.dtype,np.integer) or g.min()<0 or g.max()>self.base.P:
            raise ValueError('invalid encoded grid')
        d=self.describe(g); payload=d.pop('_description_bytes'); crop,box=self.tight_crop(g)
        Q,B=self.ambiguity(crop); ref=self.reference(); L=d['description_bits']; Lr=ref['description']['description_bits']
        ratio=max(0.,(L-self.single_bits)/(Lr-self.single_bits))
        adjusted=ratio*(1+self.config.confusion_weight*Q)/(1+self.config.confusion_weight*ref['Q'])
        def display(x): return int(math.floor(1+9999*min(1.,x)**self.config.score_exponent+0.5))
        n=int(np.count_nonzero(crop)); used,counts=np.unique(crop[crop>0],return_counts=True); p=counts/n
        result={'model_version':VERSION,'structural_score':display(ratio),
                'craft_adjusted_complexity_score':display(adjusted),'structural_ratio':ratio,
                'craft_adjusted_ratio':adjusted,'above_reference':adjusted>1,
                'occupied_beads_N':n,'active_area_A':int(crop.size),'used_colors_K':len(used),
                'global_color_entropy_bits':float(-np.sum(p*np.log2(p))),
                'color_ambiguity_Q':Q,'boundary_density_B':B,'description':d,
                'reference_description_bits':Lr,'singleton_description_bits':self.single_bits,
                'reference_Q':ref['Q'],'reference_grid_sha256':ref['grid_sha256'],
                'palette_sha256':self.base.palette_sha256,'config':asdict(self.config),
                'runtime':{'python':platform.python_version(),'numpy':np.__version__, 'zlib':zlib.ZLIB_RUNTIME_VERSION},
                'encoded_grid_sha256':hashlib.sha256(crop.astype('<i2').tobytes()).hexdigest()}
        if include_v1_workload:
            v1=self.base.score(g,encoded=True)
            result['workload_v1']={k:v1[k] for k in ('difficulty_score','raw_workload_W','workload_per_bead')}
        if keep_description: result['_description_bytes']=payload
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid',type=Path,required=True)
    p.add_argument('--palette',type=Path,default=Path(__file__).with_name('MARD_221_palette.json'))
    p.add_argument('--config',type=Path)
    p.add_argument('--out',type=Path,default=Path('complexity_results.json'))
    p.add_argument('--description-out',type=Path,help='optional lossless binary description (.bpc)')
    p.add_argument('--include-workload',action='store_true',help='also compute the separate original v1 workload')
    a=p.parse_args(); cfg=Config(**json.loads(a.config.read_text())) if a.config else Config()
    model=ComplexityModel(json.loads(a.palette.read_text()),cfg)
    result=model.score(read_code_matrix(a.grid),keep_description=bool(a.description_out),include_v1_workload=a.include_workload)
    if a.description_out:
        a.description_out.parent.mkdir(parents=True,exist_ok=True); a.description_out.write_bytes(result.pop('_description_bytes'))
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(f"Structural complexity {result['structural_score']:,}/10,000; "
          f"craft-adjusted complexity {result['craft_adjusted_complexity_score']:,}/10,000; "
          f"{result['occupied_beads_N']:,} beads. Not calibrated to time.")
    print(a.out)

if __name__=='__main__': main()
