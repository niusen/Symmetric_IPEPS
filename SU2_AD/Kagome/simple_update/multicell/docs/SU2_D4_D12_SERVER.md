# Pure Julia/TensorKit kagome D4..12 server job

Run `sh run_su2_D4_D12_server.sh` in this directory. No Python or Slurm is
called. All options are at the top of `run_su2_D4_D12_server.jl`.
The sh starts one Julia process group and prints the log / stop command.
The Julia entry selects the CPU or GPU project and installs missing Julia
dependencies on the first run. There is no supervisor or auto-restart loop.

Defaults: sqrt3xsqrt3, h=0.38, Dmax=4..12, 10 CPU/BLAS threads, chi40,
`device="cpu"`, CTMRG tolerance1e-6, at most50 iterations, and
`measurement_representation=:u1`. Choose `:dense` for ordinary TensorKit
ComplexSpace tensors. GPU remains optional: set `device="cuda:1"` explicitly.

The bundled evolved SU2 D4 candidate initializes D4; each completed D
initializes D+1, without noise. Complete multiplets are retained during SU.
Dmax counts all 2j+1 states, not just multiplets; actual bond spaces and
actual dimensions are recorded separately.

| dt | tau | full steps = round(tau/dt) |
|---|---:|---:|
| .03 | 40 | 1333 |
| .01 | 20 | 2000 |
| .003 | 5 | 1667 |
| .001 | 2 | 2000 |
| .0005 | 1 | 2000 |

Every D receives exactly9000 full Trotter steps, with the original boundary
half-steps at every stage. No early stopping, extra hold, intermediate
CTMRG or finite-torus measurement is performed. One final CTMRG environment
is reused for all final observables. The HOSVD and SU kernels are unchanged.

## Native contraction and conventions

Implementation: `src/bosonic/kagome/native_measurement/`.
Existing SU2/U1 conversion and SU kernels are reused, together with the
library's CTM settings/structures, coordinate maps, four-corner projector
construction, strip update and CUDA storage helpers. Specialized factor
contractions retain separate rank3 Q/R double layers; the onsite rank4
D^8 double layer is never constructed. Only the small corner spectra are
copied to CPU for convergence diagnostics. TensorKit/CUDA/cuTENSOR/Adapt
provide the GPU contraction/storage path; the bootstrap uses `environments/gpu/Project.toml`.
If CUDA is unavailable, the job fails before SU instead of selecting CPU.

The projected auxiliary spin1/2 becomes a one-dimensional charged U1 leg.
It remains explicit so each local TensorMap is an invariant intertwiner.
It is fixed, not summed over the full SU2 multiplet. Charged S+/S- operators
also carry conjugate one-dimensional charge legs, which are contracted
between the two operator insertions. Neutral Sz, bond Hamiltonians and
SzSzSz need no operator charge leg. This prevents transverse exchange from
being dropped by a U1 restriction of Sx or Sy.

Physical convention: H=Hex+h*sum(Sz), m=-2*mean(Sz). The field is included
once. The selected auxiliary sector fixes m=1/9; this job compares energies
within that sector, rather than measuring a plateau width.

## Output and recovery

Default output is `results/su2_D4_D12_h038_chi40_native/`.
Every completed D updates summary.csv/.toml/.json/.txt, site_magnetization.csv,
bond_energies.csv (18 bonds), triangle_observables.csv (six triangle energies
and raw C=<Sz_i Sz_j Sz_k>), and virtual_spaces.csv. Per-D folders retain
initial/stage/latest/final SU2 checkpoints, projected U1 transport, actual
virtual spaces, converted measurement states, and full CTM observables in
JLD2/TOML/JSON. Dense conversion also exposes ordinary Julia tensor arrays.

Historical dense references in data/dense_reference_h038.csv are compared with
E_SU2-E_dense, retaining their chi, magnetization, initialization/source and
convergence flags. They are candidate comparisons, not an SU2 energy bound.
Unconverged CTMRG results at the50-iteration cap remain explicitly flagged.

Rerun the same sh after interruption without changing settings. Body
checkpoints every250 full steps preserve the boundary-half-step phase.
Finished CTMRG measurements are skipped; only interrupted measurements
are repeated. Input/configuration hashes prevent mixing states/settings.
Linux flock is released when the process exits, including SIGKILL.
The printed `kill -TERM -- -PID` stops the entire Julia process group.

## Standalone conversion / measurement

In examples/run_kagome_settings.jl choose task=:convert and edit conversion_options:
input, representation=:u1 or :dense, component, h, device, chi, CTM cap/tolerance,
and measure=true/false. This native infinite-system path has no D<=6 guard.
The old export_and_verify function remains for independent small finite-torus
checks only. No Python package path or Python executable is needed.
