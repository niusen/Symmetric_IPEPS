"""Summarize all completed runs, retaining nonconvergence and cutoff dependence."""
from pathlib import Path
import json,tomllib

ROOT=Path(__file__).resolve().parents[6]
OUT=ROOT/'Kagome/fermionic/iPESS/results/gfpeps_validation'


def summarize():
    rows=[]
    directories=[OUT/f'{family}_D{d}' for family in ('ipeps','ipess') for d in (4,8)]
    for directory in directories:
        family,d=directory.name.split('_D')
        for path in directory.glob('ctm_*.toml'):
            row=tomllib.loads(path.read_text())
            row.update(family=family,D=int(d),file=path.relative_to(OUT).as_posix())
            row['max_correlation_error']=max(row[k] for k in row if k.startswith('max_') and k.endswith('_error'))
            rows.append(row)
    rows.sort(key=lambda r:(r['family'],r['D'],r['chi'],r['projector_tol']))
    (OUT/'all_results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Same-state Gaussian / CTMRG comparison','',
        'All energies are per physical kagome site, in units of the hopping amplitude.',
        '`Egrand = Ekin - mu_up*n_up - mu_down*n_down`. Reference: exact Gaussian contraction of each saved state; not the exact Hamiltonian ground state.','',
        '## Gaussian references','',
        '| Family | D | Ekin | Egrand | n_up | n_down |',
        '|---|---:|---:|---:|---:|---:|']
    for directory in directories:
        path=directory/'gaussian_reference.json'
        if not path.exists():continue
        r=json.loads(path.read_text())[-1]; family,d=directory.name.split('_D')
        lines.append(f"| {family} | {d} | {r['kinetic_per_site']:.12f} | {r['grand_per_site']:.12f} | {r['n_up_per_site']:.12f} | {r['n_down_per_site']:.12f} |")
    lines+=['','## All completed CTMRG runs','',
        'The final column means that the corner-spectrum stopping criterion passed its requested tolerance. Energy and correlation errors are checked separately. Failed import prototypes are excluded and retained under `debug_*`.','',
        '| Family | D | chi | projector cutoff | double layer / update | gauge sweeps | iterations | corner error | ΔEkin | ΔEgrand | max checked correlation error | corner converged | result |',
        '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        lines.append(f"| {r['family']} | {r['D']} | {r['chi']} | {r['projector_tol']:.0e} | {r.get('double_layer_implementation','shared_original')} / {r.get('update_method','continuous_update')} | {r.get('gauge_sweeps',0)} | {r['iterations']} | {r['ctm_error']:.2e} | {r['kinetic_per_site_difference']:+.3e} | {r['grand_per_site_difference']:+.3e} | {r['max_correlation_error']:.2e} | {r['converged']} | [{r.get('label') or 'data'}]({r['file']}) |")
    lines+=['','## Independent checks','',
        '| Check | maximum covariance error |','|---|---:|']
    for path in sorted(OUT.glob('*/source.json')):
        r=json.loads(path.read_text());error=max(c['covariance_error'] for c in r['checks'])
        lines.append(f'| {path.parent.name}: every local Gaussian → Fock tensor | {error:.3e} |')
    for name in ('triangle_check.json','block_check.json'):
        for path in sorted(OUT.glob('*/'+name)):
            r=json.loads(path.read_text())
            lines.append(f"| {path.parent.name}: {name.removesuffix('.json')} | {r['covariance_error']:.3e} |")
    lines+=['',
        'The source archives are untouched. All imported tensors retain the source particle/hole basis `b=(up,down-hole)` and the evaluator transforms energies and densities back to physical spin quantities.',
        '',
        'Agreement validates the conversion, blocking and tested CTMRG observables for these states. It does not establish correctness or convergence of SU, FU, or AD optimization. Unstable low-cutoff runs must not be presented as converged results.','']
    (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':summarize()
