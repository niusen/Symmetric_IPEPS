"""Exact sparse wavefunction for equal-amplitude oriented star singlet covers.

Physical color is a two-bit digit. No tensor-network truncation is used.
The orientation agrees with Measure_star_RVB.jl: (site,L,D,R) on even
sites and (site,L,R,U) on odd sites, up to a cover-independent phase.
"""
import itertools
import json
from pathlib import Path
import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def geometry(lx, ly):
    stars = []
    for x in range(lx):
        for y in range(ly):
            i = x * ly + y
            left = ((x - 1) % lx) * ly + y
            right = ((x + 1) % lx) * ly + y
            vertical = x * ly + (y + (-1 if (x+y) % 2 == 0 else 1)) % ly
            stars.append((i, left, vertical, right) if (x+y) % 2 == 0
                         else (i, left, right, vertical))
    return stars


def coverings(stars):
    n = len(stars)
    masks = [sum(1 << j for j in s) for s in stars]
    options = [[i for i, s in enumerate(stars) if j in s] for j in range(n)]
    result = []
    def visit(mask, centers):
        if mask == (1 << n)-1:
            result.append(tuple(centers))
            return
        j = min((j for j in range(n) if not (mask >> j) & 1),
                key=lambda j: sum(not mask & masks[i] for i in options[j]))
        for i in options[j]:
            if not mask & masks[i]:
                visit(mask | masks[i], centers + [i])
    visit(0, [])
    return result


PERMS = list(itertools.permutations(range(4)))
SIGNS = np.array([(-1)**sum(p[i] > p[j] for i in range(4) for j in range(i+1, 4))
                  for p in PERMS], dtype=np.int64)


def state(stars, covers):
    codes, amplitudes = [], []
    for cover in covers:
        c = np.array([0], dtype=np.uint64)
        a = np.array([1], dtype=np.int64)
        for center in cover:
            local = np.array([sum(color << (2*j) for color, j in zip(p, stars[center]))
                              for p in PERMS], dtype=np.uint64)
            c = (c[:, None] | local[None, :]).ravel()
            a = (a[:, None] * SIGNS[None, :]).ravel()
        codes.append(c)
        amplitudes.append(a)
    c = np.concatenate(codes)
    a = np.concatenate(amplitudes)
    order = np.argsort(c)
    c, a = c[order], a[order]
    indices = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1]
    summed = np.add.reduceat(a, indices)
    keep = summed != 0
    return c[indices][keep], summed[keep].astype(float)


def lookup(codes, a, targets):
    pos = np.searchsorted(codes, targets)
    valid = pos < len(codes)
    pos = np.minimum(pos, len(codes)-1)
    valid &= codes[pos] == targets
    return np.where(valid, a[pos], 0.)


def swap(c, i, j):
    difference = ((c >> np.uint64(2*i)) ^ (c >> np.uint64(2*j))) & np.uint64(3)
    return c ^ (difference << np.uint64(2*i)) ^ (difference << np.uint64(2*j))


def measure(lx, ly):
    stars = geometry(lx, ly)
    covers = coverings(stars)
    codes, a = state(stars, covers)
    norm = np.dot(a, a)
    def permutation(pairs):
        c = codes.copy()
        for i, j in pairs:
            c = swap(c, i, j)
        return float(np.dot(a, lookup(codes, a, c)) / norm)
    bonds = sorted({tuple(sorted((s[0], j))) for s in stars for j in s[1:]})
    bond_p = [permutation([b]) for b in bonds]
    # Rank-one four-site SU(4) singlet projector. Sum epsilon-weighted
    # amplitudes with fixed external colors, then square (epsilon norm^2=24).
    singlets = []
    for s in stars:
        colors = np.stack([(codes >> np.uint64(2*j)) & np.uint64(3) for j in s], axis=1)
        valid = np.all(np.sort(colors, axis=1) == np.arange(4), axis=1)
        col = colors[valid]
        sign = (-1.)**sum((col[:, i] > col[:, j]).astype(int)
                         for i in range(4) for j in range(i+1, 4))
        mask = np.uint64(sum(3 << (2*j) for j in s))
        outside = codes[valid] & ~mask
        order = np.argsort(outside)
        outside, v = outside[order], (a[valid] * sign)[order]
        idx = np.r_[0, np.flatnonzero(outside[1:] != outside[:-1])+1]
        projected = np.add.reduceat(v, idx)
        singlets.append(float(np.dot(projected, projected)/(24*norm)))
    result = dict(Lx=lx, Ly=ly, covers=len(covers), nonzero_amplitudes=len(codes),
                  norm_over_product_norm=norm / 24**(len(stars)//4),
                  E_P_per_site=sum(bond_p)/len(stars),
                  bonds=bonds, bond_P=bond_p, star_singlet_projector=singlets,
                  P_0j=[permutation([(0,j)]) if j else 1. for j in range(len(stars))],
                  connected_bond_01=[permutation([bonds[0], b])-bond_p[0]*p
                                     for b, p in zip(bonds,bond_p)])
    print(json.dumps(result), flush=True)
    out = DATA_ROOT / 'star_RVB'
    out.mkdir(exist_ok=True)
    (out / f'exact_{lx}x{ly}.json').write_text(json.dumps(result, indent=2))
    return result


def overlap(stars, bra, ket, exchange=None, singlet=None):
    """Contract two valence-cluster states, without a physical wavefunction.

    Greedily eliminate parallel edges first; this is efficient for the
    small collection of star covers here. Reject large intermediates.
    """
    eps = np.zeros((4,)*4)
    for p, sign in zip(PERMS, SIGNS):
        eps[p] = sign / np.sqrt(24.)
    relabel = {} if singlet is None else {j:len(stars)+j for j in singlet}
    tensors = [(eps, [relabel.get(j,j) for j in stars[c]]) for c in bra]
    exchange = exchange or {}
    tensors += [(eps, [exchange.get(j,j) for j in stars[c]]) for c in ket]
    if singlet is not None:
        tensors += [(eps,[relabel[j] for j in singlet]),(eps,list(singlet))]
    def plan(seed):
        labels=[set(inds) for _,inds in tensors]
        rng=np.random.default_rng(seed)
        steps=[]; peak=0
        while labels:
            choices=[]
            for i in range(len(labels)):
                for j in range(i+1,len(labels)):
                    shared=labels[i]&labels[j]
                    if shared:
                        rank=len(labels[i])+len(labels[j])-2*len(shared)
                        score=(rank,-len(shared)) if seed==0 else (rank+rng.uniform(0,3),0)
                        choices.append((score,i,j,rank))
            if not choices:
                raise RuntimeError('unpaired physical index')
            _,i,j,rank=min(choices)
            peak=max(peak,rank);steps.append((i,j))
            remaining=labels[i]^labels[j]
            labels.pop(j);labels.pop(i)
            if remaining:labels.append(remaining)
        return peak,steps
    peak,steps=plan(0)
    if peak>12:
        for seed in range(1,65):
            candidate=plan(seed)
            if candidate[0]<peak:peak,steps=candidate
            if peak<=12:break
    if peak>12:
        raise RuntimeError(f'contraction intermediate rank {peak} is too large')
    scalar = 1.
    for i,j in steps:
        a, ia = tensors[i]; b, ib = tensors[j]
        shared = sorted(set(ia)&set(ib))
        c = np.tensordot(a,b,axes=([ia.index(k) for k in shared],
                                  [ib.index(k) for k in shared]))
        ic = [k for k in ia if k not in shared]+[k for k in ib if k not in shared]
        tensors.pop(j);tensors.pop(i)
        if ic:
            tensors.append((c,ic))
        else:
            scalar *= float(c)
    return scalar


def measure_overlaps(lx, ly, extra=False):
    stars = geometry(lx,ly); covers = coverings(stars)
    # Translation invariance and horizontal reflection leave two bond types
    # on these rectangular tori: vertical and horizontal.
    bonds = [(0,stars[0][1]),(0,stars[0][2])]
    sums = np.zeros(6 if extra else 3)
    far = (lx//2)*ly
    farbond = (far, stars[far][2])  # even parity for the sizes used here
    for i,bra in enumerate(covers):
        for j in range(i,len(covers)):
            ket = covers[j]; factor = 1 if i==j else 2
            sums[0] += factor*overlap(stars,bra,ket)
            for k,(u,v) in enumerate(bonds):
                sums[k+1] += factor*overlap(stars,bra,ket,{u:v,v:u})
            if extra:
                sums[3] += factor*overlap(stars,bra,ket,singlet=stars[0])
                u,v=bonds[1];s,t=farbond
                sums[4] += factor*overlap(stars,bra,ket,{u:v,v:u,s:t,t:s})
                sums[5] += factor*overlap(stars,bra,ket,{0:far,far:0})
    ph,pv = sums[1:3]/sums[0]
    result = dict(Lx=lx,Ly=ly,covers=len(covers),norm_over_product_norm=sums[0],
                  P_horizontal=ph,P_vertical=pv,E_P_per_site=ph+pv/2)
    if extra:
        result.update(star_singlet_projector=sums[3]/sums[0],
                      far_vertical_bond=farbond,
                      connected_vertical_bonds=sums[4]/sums[0]-pv*pv,
                      far_P=sums[5]/sums[0],
                      far_SU4_generator_correlation=(sums[5]/sums[0]-.25)/2)
    print(json.dumps(result),flush=True)
    out = DATA_ROOT/'star_RVB';out.mkdir(exist_ok=True)
    (out/f'overlap_{lx}x{ly}.json').write_text(json.dumps(result,indent=2))
    return result


def compare_evolved_star_weights():
    """Compare normalized covering coefficients after bond-gauge alignment.

    Run Inspect_star_channels.jl first. Its residual check verifies that the
    evolved D=9 tensors remain scalar multiples of each original star channel.
    """
    import csv
    import cmath
    folder = DATA_ROOT/'star_SU_D9_L4'
    coefficients = {}
    with (folder/'star_channel_coefficients.csv').open() as stream:
        for row in csv.DictReader(stream):
            key = (int(row['x'])-1, int(row['y'])-1,
                   tuple(int(row[f'q{k}']) for k in (1,2,3)))
            coefficients[key] = complex(float(row['re']),float(row['im']))
    results = []
    for lx,ly in [(4,4),(8,4),(8,8),(12,8)]:
        stars = geometry(lx,ly)
        covers = coverings(stars)
        log_amplitudes = []
        for cover in covers:
            centers = set(cover)
            amplitude = 0j
            for i,star in enumerate(stars):
                q = (1,1,1) if i in centers else tuple(3 if j in centers else 0 for j in star[1:])
                x,y = divmod(i,ly)
                amplitude += cmath.log(coefficients[x%4,y%4,q])
            log_amplitudes.append(amplitude)
        ratios = [cmath.exp(z-log_amplitudes[0]) for z in log_amplitudes]
        row = dict(Lx=lx,Ly=ly,covers=len(covers),
                   max_relative_cover_weight_change=max(abs(r-1) for r in ratios),
                   log_abs_overall_scale=log_amplitudes[0].real)
        results.append(row)
        print(json.dumps(row),flush=True)
    (folder/'cover_weight_comparison.json').write_text(json.dumps(results,indent=2))
    return results


if __name__ == '__main__':
    import sys
    if '--compare-evolved' in sys.argv:
        compare_evolved_star_weights()
    elif '--observables' in sys.argv:
        measure_overlaps(8,8,extra=True)
    elif '--overlap' in sys.argv:
        for shape in [(4,2),(4,4),(8,4),(12,4),(8,8)]:
            measure_overlaps(*shape)
    else:
        measure(4, 2)
        measure(4, 4)
