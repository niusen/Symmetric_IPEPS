"""Read native dense Juraj JSON and use Juraj's unmodified CTMRG/model/RDM.

The input wavefunction was obtained with Julia U1 tensors. No U1 backend,
Julia contraction, simple update, or optimization is used in this measurement.
Server settings are supplied by run_server.py; a JSON plan is also accepted.
"""
from pathlib import Path
from contextlib import contextmanager
import ast
import csv
import hashlib
import json
import os
import sys
import time
import tomllib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@contextmanager
def output_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'writer.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if handle.seek(0, 2) == 0:
                handle.write(b' ')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f'Another live Juraj measurement owns {folder}') from exc
            unlock = lambda: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Another live Juraj measurement owns {folder}') from exc
            unlock = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            yield
        finally:
            unlock()


def configure(plan):
    repo = Path(plan['juraj_repository']).resolve()
    if not (repo / 'ipeps' / 'ipess_kagome.py').is_file():
        raise FileNotFoundError(f'Juraj repository not found: {repo}')
    os.environ['OMP_NUM_THREADS'] = str(plan['threads'])
    os.environ['MKL_NUM_THREADS'] = str(plan['threads'])
    sys.path.insert(0, str(repo))
    import torch
    import config as cfg
    torch.set_num_threads(plan['threads'])
    cfg.global_args.dtype = 'complex128'
    cfg.global_args.torch_dtype = torch.complex128
    cfg.global_args.device = plan['device']
    cfg.ctm_args.ctm_max_iter = plan['ctm_iterations']
    cfg.ctm_args.ctm_conv_tol = plan['ctm_tolerance']
    cfg.ctm_args.projector_svd_reltol = plan['projector_cutoff']
    cfg.ctm_args.ctm_force_dl = False  # Juraj's existing single-layer CTM moves.
    cfg.ctm_args.ctm_warmup_iter = -1  # The requested iteration cap includes all sweeps.
    cfg.ctm_args.fwd_checkpoint_move = False
    files = ('ipeps/ipess_kagome.py', 'ipeps/ipeps.py', 'ctm/generic/ctmrg.py',
             'ctm/pess_kagome/rdm_kagome.py', 'models/spin_half_kagome.py')
    return torch, cfg, {name: sha(repo / name) for name in files}


def read_and_verify(state_file, torch, cfg):
    from ipeps.ipess_kagome import read_ipess_kagome_generic
    raw = json.loads(Path(state_file).read_text(encoding='utf-8'))
    # Juraj's legacy reader uses eval for coordinate keys. Accept numeric tuple
    # literals only before passing our exported state to that native reader.
    for coord in raw['ipess_tensors']:
        parsed = ast.literal_eval(coord)
        if not (isinstance(parsed, tuple) and len(parsed) == 2
                and all(type(v) is int and 0 <= v < 3 for v in parsed)):
            raise ValueError('Invalid coordinate in dense state JSON')
    state = read_ipess_kagome_generic(str(state_file))
    errors = []
    for coord, site in state.sites.items():
        t = state.ipess_tensors[coord]
        # Independently reconstruct the Julia single-site PESS tensor, then
        # rotate coordinates and physical order into Juraj's convention.
        b1 = t['B_a'].permute(2, 1, 0)
        b2 = t['B_c'].permute(2, 1, 0)
        b3 = t['B_b'].permute(2, 1, 0)
        up = t['T_d'].permute(2, 0, 1)
        down = t['T_u'].permute(2, 0, 1)
        expected = torch.einsum('lia,mjb,ukc,ijk,rmd->ldruabc', b1, b2, b3, up, down)
        expected = expected.permute(5, 6, 4, 1, 2, 3, 0)
        expected = expected.reshape(8, *expected.shape[3:])
        expected = expected / expected.abs().max()
        errors.append(float((site - expected).abs().max().item()))
    if max(errors) > 1e-12:
        raise AssertionError(f'Onsite tensor mapping mismatch: {errors}')
    return state, max(errors)


def reference_data(path):
    if not path or not Path(path).is_file():
        return None, {}
    with Path(path).open('rb') as f:
        data = tomllib.load(f)
    grouped = {}
    for cell in data['cells']:
        grouped.setdefault(cell['representative'], []).append(cell)
    result = {}
    for rep, cells in grouped.items():
        result[rep] = {key: [sum(c[key][i] for c in cells) / len(cells) for i in range(3)]
                       for key in ('bonds_up', 'bonds_down', 'Sz')}
    return data, result


def measure_point(plan, entry, chi, torch, cfg, fingerprints):
    from ipeps.ipess_kagome import read_ipess_kagome_generic
    from ctm.generic.env import ENV, init_env
    from ctm.generic import ctmrg
    from models.spin_half_kagome import S_HALF_KAGOME
    D = entry['D']
    folder = Path(plan['out']) / f'D{D}' / f'chi{chi}'
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / 'result.json'
    identity = dict(D=D, chi=chi, input_sha256=sha(entry['state_json']),
                    h=plan['h'], threads=plan['threads'], device=plan['device'],
                    ctm_iterations=plan['ctm_iterations'], ctm_tolerance=plan['ctm_tolerance'],
                    projector_cutoff=plan['projector_cutoff'], source_hashes=fingerprints,
                    bridge_sha256=sha(__file__), bridge_schema=2)
    if completed.is_file():
        result = json.loads(completed.read_text())
        if result['settings'] != identity:
            raise RuntimeError('Measurement inputs/settings changed; use a new output folder')
        print(f'SKIP completed Juraj D={D} chi={chi}', flush=True)
        return result
    start = time.perf_counter()
    state, mapping_error = read_and_verify(entry['state_json'], torch, cfg)
    # Native JSON writer -> reader round-trip, without changing coefficients.
    roundtrip = folder / 'juraj_roundtrip.json'
    state.write_to_file(str(roundtrip), normalize=False)
    reread = read_ipess_kagome_generic(str(roundtrip))
    roundtrip_error = max(float((t - reread.ipess_tensors[c][name]).abs().max().item())
                          for c, ts in state.ipess_tensors.items() for name, t in ts.items())
    if roundtrip_error > 1e-14:
        raise AssertionError('Juraj JSON round-trip changed coefficients')
    print(json.dumps(dict(event='start', D=D, chi=chi, independent_cells=len(state.sites),
                          mapping_error=mapping_error, roundtrip_error=roundtrip_error)), flush=True)
    # Measure the exchange Hamiltonian with Juraj. Add +h<Sz> once afterwards;
    # no Julia Hamiltonian or energy-measurement function is imported here.
    model = S_HALF_KAGOME(j1=1., h=0.)

    @torch.no_grad()
    def convergence(state, env, history, ctm_args):
        history = history or {'previous': None, 'iterations': 0, 'conv_crit': []}
        spectra = {k: torch.linalg.svdvals(c) for k, c in env.C.items()}
        spectra = {k: s / s[0] for k, s in spectra.items()}
        error = float('inf')
        if history['previous'] is not None:
            differences = []
            for key, values in spectra.items():
                old = history['previous'][key]
                delta = values.new_zeros(max(values.numel(), old.numel()))
                delta[:values.numel()] += values
                delta[:old.numel()] -= old
                differences.append(float(delta.norm().item()))
            error = max(differences)
        history.update(previous=spectra, iterations=history['iterations'] + 1,
                       error=error, converged=error < plan['ctm_tolerance'])
        history['conv_crit'].append(error)
        print(json.dumps(dict(event='ctm', D=D, chi=chi, iteration=history['iterations'],
                              error=error, seconds=time.perf_counter() - start)), flush=True)
        return history['converged'], history

    with torch.no_grad():
        print(f'INITIALIZE Juraj D={D} chi={chi}', flush=True)
        env = ENV(chi, state)
        init_env(state, env)
        env, history, *_ = ctmrg.run(state, env, conv_check=convergence)
        print(f'OBSERVABLES Juraj D={D} chi={chi}', flush=True)
        values, labels = model.eval_obs(state, env, force_cpu=False, cast_real=True)
    raw_obs = dict(zip(labels, values))
    def number(label):
        value = raw_obs[label]
        return complex(value.item() if hasattr(value, 'item') else value)
    max_imag = max(abs(number(k).imag) for k in raw_obs)
    ref, ref_groups = reference_data(entry.get('julia_reference'))
    ref_chi = ref['settings']['chi'] if ref else None
    bonds, sites, triangles = [], [], []
    exchange_sum = 0.
    # Onsite physical order is (s0,s1,s2)=(Julia B2,B3,B1).
    # IMPORTANT: rdm2x2_up_triangle_open contracts RU first, then LD and RD;
    # its output physical order is (RU.s1,LD.s2,RD.s0), NOT (s0,s1,s2).
    # These are (Julia B3(R+ey),B1(R+ex),B2(R)). The diagonal anchor
    # shift preserves sqrt3 representative (x-y)%3. No tensor is reordered
    # or energy modified here: this only identifies the bond labels.
    bonds_map = (
        ('dn', 'up', '02', 0, 'B1@R', 'B2@R'),
        ('dn', 'up', '01', 1, 'B2@R', 'B3@R'),
        ('dn', 'up', '12', 2, 'B3@R', 'B1@R'),
        ('up', 'down', '02', 0, 'B2@R', 'B3@R+ey'),
        ('up', 'down', '01', 1, 'B3@R+ey', 'B1@R+ex'),
        ('up', 'down', '12', 2, 'B1@R+ex', 'B2@R'),
    )
    for x, y in state.sites:
        suffix = f'({x},{y})'
        rep = f'{1 + (y - x) % 3},1'
        for juraj_orientation, julia_orientation in (('dn', 'up'), ('up', 'down')):
            energy = number(f'e_t_{juraj_orientation}{suffix}').real
            exchange_sum += energy
            triangles.append(dict(D=D, chi=chi, representative=rep, orientation=julia_orientation, energy=energy))
        for orientation, julia_orientation, pair, index, va, vb in bonds_map:
            energy = number(f'SS_{orientation}_{pair}{suffix}').real
            expected = ref_groups.get(rep, {}).get(f'bonds_{julia_orientation}', [None] * 3)[index]
            bonds.append(dict(D=D, chi=chi, representative=rep, orientation=julia_orientation,
                              vertex_a=va, vertex_b=vb, energy=energy,
                              julia_reference_chi=ref_chi, same_chi=(ref_chi == chi),
                              julia_reference_energy=expected,
                              delta_juraj_minus_julia=None if expected is None else energy - expected))
        for index, b in ((0, 2), (1, 3), (2, 1)):
            z = number(f'sz_{index}{suffix}').real
            sites.append(dict(D=D, chi=chi, representative=rep, site=f'B{b}', Sz=z, m_normalized=-2 * z))
    nspins = 3 * len(state.sites)
    if len(bonds) != 18 or len(sites) != 9:
        raise AssertionError('Unexpected number of independent kagome bonds/sites')
    exchange = exchange_sum / nspins
    mean_sz = sum(r['Sz'] for r in sites) / nspins
    summary = dict(D=D, chi=chi, energy_per_spin=exchange + plan['h'] * mean_sz,
                   exchange_energy_per_spin=exchange, field_energy_per_spin=plan['h'] * mean_sz,
                   m_normalized=-2 * mean_sz, ctm_converged=history['converged'],
                   ctm_iterations=history['iterations'], ctm_error=history['error'],
                   seconds=time.perf_counter() - start, mapping_error=mapping_error,
                   roundtrip_error=roundtrip_error, max_observable_imaginary_part=max_imag,
                   triangle_vs_bond_sum_residual=exchange - sum(r['energy'] for r in bonds) / nspins,
                   julia_reference_chi=ref_chi, same_chi=(ref_chi == chi),
                   julia_reference_total=None if ref is None else ref['energy_per_spin'],
                   delta_total_juraj_minus_julia=None if ref is None else exchange + plan['h'] * mean_sz - ref['energy_per_spin'],
                   max_bond_delta_juraj_minus_julia=None if ref is None else max(abs(r['delta_juraj_minus_julia']) for r in bonds))
    result = dict(settings=identity, summary=summary, bonds=bonds, sites=sites, triangles=triangles,
                  raw_observables={k: dict(real=number(k).real, imag=number(k).imag) for k in raw_obs})
    write_csv(folder / 'bond_energies.csv', bonds)
    write_csv(folder / 'site_magnetization.csv', sites)
    atomic_json(completed, result)
    print(json.dumps(dict(event='final', **summary)), flush=True)
    return result


def run_plan(plan):
    if not (0 < plan['threads'] and 2 <= plan['ctm_iterations'] <= 50 and plan['ctm_tolerance'] > 0):
        raise ValueError('Invalid measurement settings')
    torch, cfg, fingerprints = configure(plan)
    with output_lock(plan['out']):
        results = []
        for entry in plan['states']:
            for chi in plan['chis']:
                result = measure_point(plan, entry, chi, torch, cfg, fingerprints)
                results.append(result)
                root = Path(plan['out'])
                write_csv(root / 'summary.csv', [r['summary'] for r in results])
                write_csv(root / 'bond_energies.csv', [b for r in results for b in r['bonds']])
                write_csv(root / 'site_magnetization.csv', [s for r in results for s in r['sites']])
                atomic_json(root / 'summary.json', [r['summary'] for r in results])
                if plan['device'].startswith('cuda'):
                    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Run sh run.sh; server settings are in run_server.py')
    run_plan(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
