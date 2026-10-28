"""Forward-only block-U1 or dense measurement of explicitly polarized PESS."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import tomllib

sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ipeps-root', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--h', type=float, default=.38)
    p.add_argument('--chi', type=int, nargs='+', default=[8])
    p.add_argument('--threads', type=int, default=10)
    p.add_argument('--ctm-iterations', type=int, default=100)
    p.add_argument('--ctm-tol', type=float, default=1e-6)
    a = p.parse_args()
    if not 2 <= a.ctm_iterations <= 100 or min(a.chi) < 1 or a.threads < 1:
        p.error('require 2..100 CTM iterations, positive chi and threads')
    if not math.isfinite(a.h) or not math.isfinite(a.ctm_tol) or a.ctm_tol <= 0:
        p.error('require finite h and positive finite CTM tolerance')
    if not (a.ipeps_root / 'ctmrg' / 'kagome_ctmrg.py').is_file():
        p.error('--ipeps-root must be the existing iPEPS_Z2 package directory')
    sys.path.insert(0, str(a.ipeps_root.resolve()))
    if (a.ipeps_root / 'vendor').is_dir():
        sys.path.insert(0, str((a.ipeps_root / 'vendor').resolve()))
    import torch
    from ansatz.kagome_iPESS import load_kagome_iPESS
    from ctmrg.kagome_ctmrg import kagome_ctm_settings, ctmrg_kagome
    from model.kagome_spin import kagome_energy

    torch.set_num_threads(a.threads)
    if a.device.startswith('cuda'):
        index = torch.device(a.device).index or 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            p.error(f'{a.device} unavailable; select an existing device explicitly')
    if a.out.exists():
        p.error('--out already exists; choose a new measurement directory')
    a.out.mkdir(parents=True)
    raw = a.state.read_bytes()
    data = tomllib.loads(raw.decode('utf-8')) if a.state.suffix == '.toml' else json.loads(raw)
    is_u1 = data.get('format') == 'kagome_u1_portable_v1'
    canonical = a.out / ('u1_state.json' if is_u1 else 'dense_state.json')
    canonical.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
    if is_u1:
        from u1_ctm_backend import U1State, ctmrg_u1, u1_energy
        state = U1State(data, a.device)
    else:
        state = load_kagome_iPESS(canonical, device=a.device)
    settings = dict(h=a.h, D=state.D, device=a.device, threads=a.threads,
                    ctm_tol=a.ctm_tol, ctm_iterations=a.ctm_iterations,
                    input_sha256=hashlib.sha256(raw).hexdigest(),
                    source_checkpoint_sha256=data.get('conversion', {}).get('source_sha256'),
                    symmetry='U1' if is_u1 else 'dense',
                    method='infinite lattice, block U1 Q/R CTMRG' if is_u1 else 'infinite lattice, dense Q/R CTMRG')
    rows = []
    with torch.no_grad():
        for chi in a.chi:
            start = time.perf_counter()
            cfg = kagome_ctm_settings(chi, a.ctm_iterations, a.ctm_tol)
            cfg.CTM_trun_tol = 1e-12
            cfg.checkpoint_move = cfg.checkpoint_energy = cfg.use_sub_checkpoint = False
            cfg.CTM_ite_info = True
            env, left, right, info = ctmrg_u1(state,cfg) if is_u1 else ctmrg_kagome(state,cfg)
            if is_u1:
                energy,obs = u1_energy(state,env,left,right,a.h)
            else:
                energy, obs = kagome_energy(state, env, left, right, h_z=a.h,
                    checkpoint_energy=False, return_observables=True)
            mz = obs['Sz'].mean().item()
            exchange = ((obs['E_up']+obs['E_down']).mean()/3).item()
            field = (obs['E_field'].mean()/3).item()
            error = abs(energy.item()-exchange-a.h*mz)
            if error > 1e-10 or abs(field-a.h*mz) > 1e-10:
                raise AssertionError('energy / onsite field reconstruction failed')
            if not all(math.isfinite(x) for x in (energy.item(), exchange, mz)):
                raise FloatingPointError('nonfinite observable')
            sites = {}
            max_spread = 0.
            for rep in state.inequivalent_cells:
                values = torch.stack([obs['Sz'][x-1,y-1] for x in range(1,state.Lx+1)
                                     for y in range(1,state.Ly+1) if state.site_map[f'{x},{y}'] == rep])
                average = values.mean(0)
                max_spread = max(max_spread, (values-average).abs().max().item())
                sites[rep] = average.cpu().tolist()
            row = dict(settings=settings, chi=chi, ctm=info, energy_per_spin=energy.item(),
                       exchange_energy_per_spin=exchange, field_energy_per_spin=field,
                       mean_Sz=mz, m_normalized=-2*mz, distance_to_one_ninth=-2*mz-1/9,
                       site_Sz=sites, equivalent_position_max_spread=max_spread,
                       energy_reconstruction_error=error,
                       actual_corner_dimensions=sorted({t.get_shape()[0]
                           for cell in env['Cset'].values() for t in cell.values()}),
                       seconds=time.perf_counter()-start)
            (a.out / f'chi_{chi}.json').write_text(json.dumps(row, indent=2, allow_nan=False))
            rows.append(row)
            print(('U1_CTM ' if is_u1 else 'DENSE_CTM ')+json.dumps(row, allow_nan=False), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
