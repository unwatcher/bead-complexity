#!/usr/bin/env python3
"""Reproduce order-aware scores and matched controls; checks are not human validation."""
from pathlib import Path
import csv, json, math
import numpy as np
from bead_complexity import ComplexityModel, Config, read_code_matrix, transform

HERE=Path(__file__).resolve().parent

def save_matrix(path,g):
    with path.open('w',newline='',encoding='utf-8') as f:
        out=csv.writer(f); out.writerow(['row']+list(range(1,g.shape[1]+1)))
        for r,row in enumerate(g): out.writerow([r+1]+row.tolist())

def main():
    model=ComplexityModel(json.loads((HERE/'MARD_221_palette.json').read_text()))
    (HERE/'results').mkdir(exist_ok=True)
    s=read_code_matrix(HERE/'example_inputs/seated_girl_latest.csv')
    a=read_code_matrix(HERE/'example_inputs/amber_hat_latest.csv')
    m=read_code_matrix(HERE/'example_inputs/moonlit_zoom_latest.csv')
    yy,xx=np.indices(s.shape); checker=np.where((yy+xx)%2,'H7','H2')
    near=np.where((yy+xx)%2,'D2','H15')
    mask=model.base.encode(s)>0
    checker_masked=np.where(mask,checker,'')
    used=sorted(set(s[mask]),key=lambda x:model.base.code_to_id[x])
    # Exactly the same occupied cells AND all 24 source colors, in a regular 4x6 tile.
    motif=np.array(used).reshape(4,6)
    tiled=np.where(mask,motif[yy%4,xx%6],'')
    rng=np.random.default_rng(20260919)
    shuffled=s.copy(); shuffled[mask]=rng.permutation(shuffled[mask])
    shuffled_full=s.ravel()[rng.permutation(s.size)].reshape(s.shape)
    corrupt=checker.copy(); indices=rng.choice(checker.size,size=500,replace=False)
    corrupt.ravel()[indices]=np.where(corrupt.ravel()[indices]=='H7','H2','H7')
    almost=checker.copy(); almost[40,61]='H2' if checker[40,61]=='H7' else 'H7'
    g221=np.array(model.base.codes)
    examples={
        'single_black':np.array([['H7']]),
        'solid_black_100x100':np.full(s.shape,'H7'),
        'checkerboard_100x100':checker,
        'checkerboard_similar_shades_100x100':near,
        'checkerboard_one_exception':almost,
        'checkerboard_5_percent_exceptions':corrupt,
        'checkerboard_same_seated_mask':checker_masked,
        'regular_24_color_tile_same_seated_mask':tiled,
        'seated_girl_latest':s,
        'seated_shuffled_same_mask_and_histogram':shuffled,
        'seated_shuffled_full_box_same_histogram':shuffled_full,
        'amber_hat_latest':a,
        'moonlit_zoom_latest':m,
        'random_221_100x100':g221[rng.integers(0,221,size=(100,100))],
        'solid_black_500x500':np.full((500,500),'H7'),
        'checkerboard_500x500':np.where(np.indices((500,500)).sum(axis=0)%2,'H7','H2'),
    }
    # v1 workload remains independent; compute old anchor once for all samples.
    oldreference=model.base.reference()
    rows=[]; results={}
    for name,g in examples.items():
        r=model.score(g,keep_description=True)
        blob=r.pop('_description_bytes'); (HERE/'results'/f'{name}.bpc').write_bytes(blob)
        old=model.base.score(g)
        r['workload_v1']={k:old[k] for k in ('difficulty_score','raw_workload_W','workload_per_bead')}
        (HERE/'results'/f'{name}.json').write_text(json.dumps(r,indent=2,allow_nan=False))
        results[name]=r
        row={'pattern':name,'occupied_beads':r['occupied_beads_N'],'active_area':r['active_area_A'],
             'colors':r['used_colors_K'],'description_bits':r['description']['description_bits'],
             'bits_per_active_cell':r['description']['bits_per_active_cell'],
             'v2_structural_score':r['structural_score'],
             'v2_craft_adjusted_complexity':r['craft_adjusted_complexity_score'],
             'ambiguity_Q':r['color_ambiguity_Q'],
             'v1_total_workload_score':old['difficulty_score'],
             'v1_work_units_per_bead':old['workload_per_bead'],'selected_model':r['description']['model']}
        rows.append(row); print(name,row['description_bits'],row['v2_structural_score'],row['v2_craft_adjusted_complexity'],row['v1_total_workload_score'],flush=True)
        if 'same_' in name or name.startswith('checkerboard_'):
            save_matrix(HERE/'example_inputs'/f'{name}.csv',g)
    reference_grid=model.base.benchmark_grid()
    ref=model.score(reference_grid,encoded=True)
    (HERE/'results/random_500x500_reference.json').write_text(json.dumps(ref,indent=2))
    rows.append({'pattern':'random_221_500x500_ANCHOR','occupied_beads':250000,'active_area':250000,'colors':221,
                 'description_bits':ref['description']['description_bits'],
                 'bits_per_active_cell':ref['description']['bits_per_active_cell'],
                 'v2_structural_score':ref['structural_score'],
                 'v2_craft_adjusted_complexity':ref['craft_adjusted_complexity_score'],
                 'ambiguity_Q':ref['color_ambiguity_Q'],'v1_total_workload_score':10000,
                 'v1_work_units_per_bead':oldreference['raw_workload_W']/250000,
                 'selected_model':ref['description']['model']})
    with (HERE/'comparison_scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value: raise AssertionError(name)
    check('lower_anchor_1',results['single_black']['structural_score']==1)
    check('upper_anchor_10000',ref['structural_score']==ref['craft_adjusted_complexity_score']==10000)
    check('girl_above_full_checker',results['seated_girl_latest']['structural_score']>results['checkerboard_100x100']['structural_score'])
    check('girl_above_masked_checker',results['seated_girl_latest']['structural_score']>results['checkerboard_same_seated_mask']['structural_score'])
    check('girl_above_same_mask_same_24_color_motif',results['seated_girl_latest']['structural_score']>results['regular_24_color_tile_same_seated_mask']['structural_score'])
    check('shuffle_same_mask_same_histogram_harder',results['seated_shuffled_same_mask_and_histogram']['structural_score']>results['seated_girl_latest']['structural_score'])
    check('near_color_same_structural_bits',results['checkerboard_100x100']['description']['description_bits']==results['checkerboard_similar_shades_100x100']['description']['description_bits'])
    check('near_color_higher_craft_complexity',results['checkerboard_similar_shades_100x100']['craft_adjusted_complexity_score']>results['checkerboard_100x100']['craft_adjusted_complexity_score'])
    check('one_exception_increases_bits',results['checkerboard_one_exception']['description']['description_bits']>results['checkerboard_100x100']['description']['description_bits'])
    check('5pct_exceptions_increase_bits',results['checkerboard_5_percent_exceptions']['description']['description_bits']>results['checkerboard_one_exception']['description']['description_bits'])
    g=model.base.encode(s); d=model.describe(g)
    for t in range(8):
        check(f'orientation_{t}_bit_invariance',model.describe(transform(g,t))['description_bits']==d['description_bits'])
    padded=np.pad(g,((3,11),(7,5)))
    check('outer_padding_invariance',model.describe(padded)['description_bits']==d['description_bits'])
    ids=np.unique(g[g>0]); perm=ids[::-1]; recolor=g.copy()
    for i,j in zip(ids,perm): recolor[g==i]=j
    check('palette_label_permutation_invariance',model.describe(recolor)['description_bits']==d['description_bits'])
    check('input_grid_unchanged',np.array_equal(model.base.encode(s),g))
    check('all_reported_roundtrips_exact',all(r['description']['round_trip_exact'] for r in results.values()))
    check('same_mask_control_N_matches',results['checkerboard_same_seated_mask']['occupied_beads_N']==results['seated_girl_latest']['occupied_beads_N'])
    check('24_color_motif_control_K_matches',results['regular_24_color_tile_same_seated_mask']['used_colors_K']==results['seated_girl_latest']['used_colors_K'])
    check('shuffle_preserves_histogram',sorted(shuffled[mask].tolist())==sorted(s[mask].tolist()))
    (HERE/'tests.json').write_text(json.dumps({'checks':checks,'passed':sum(checks.values()),'total':len(checks),'human_validation':False},indent=2))
    (HERE/'benchmark.json').write_text(json.dumps({'v2':model.reference(),'v1_workload':oldreference['raw_workload_W'],'single_bits':model.single_bits},indent=2))
    print(f'{len(checks)} checks passed.',flush=True)

if __name__=='__main__': main()
