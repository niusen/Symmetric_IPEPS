# Julia U1 conversion, simple update and measurement

## D4 SU2 seed -> U1 Simple Update trial

Open `examples/run_u1_from_su2.jl` and run it directly in Julia. Edit its top
settings block; no command-line options are required. It reads the bundled,
already evolved D4 SU2 candidate from `data/su2_D4_seed.toml`, selects auxiliary
component 1 once, and then calls the existing `to_u1` and `itebd_u1` functions.
The input can also be a charged SU2 `.jld2` checkpoint, including a server D4
`final.jld2`. The original source checkpoint is never overwritten.

The short local default is D=4, h=0.38, CPU/10 threads, chi=16, at most 50 CTMRG
iterations, tolerance 1e-6, and 600 full SU steps in three stages:
(dt,tau)=(0.01,2.0),(0.003,0.6),(0.001,0.2). This is a trial, not a fully
converged optimization. CTMRG runs only before and after the entire evolution.
U1 truncation keeps individual charge states rather than whole SU2 multiplets.

Each invocation creates a new timestamped directory under `results/`. It saves
initial/final U1 JLD2 checkpoints, stage checkpoints, portable U1 tensors,
initial/final virtual spaces, site/bond/triangle measurements, and
`comparison.toml` / `comparison.csv` with energies and CTMRG convergence flags.
The workflow function is in `src/bosonic/kagome/workflows/u1_from_su2_workflow.jl`;
the numerical SU and CTMRG kernels are reused without modifications.


Use examples/run_kagome_settings.jl and edit the options in the file; no command-line
parameters are needed. task=:charged_trial / :simple_update retain the existing
SU2 and U1 simple-update kernels and initial-state conventions.

For task=:convert, set conversion_options.input to a SU2 or U1 JLD2 checkpoint.
representation=:u1 is the default; :dense creates dense TensorKit tensors and
ordinary Julia arrays. component=1 selects the auxiliary highest weight.
measure=true computes infinite-system CTMRG and all observables in Julia;
measure=false only converts/saves. CPU is the default; select device="cuda:1"
explicitly for GPU. Select chi and
CTMRG tolerance/cap in this same options block. No Python subprocess is used.

The fixed one-dimensional auxiliary charge is retained in U1 TensorMaps.
Restriction is exact: virtual multiplicities and arrows are not truncated.
The physical convention is H=Hex+h*sum(Sz), m=-2*mean(Sz), q=2Sz.
Magnetization1/9 is the selected auxiliary sector and does not establish a
finite plateau width. The field term is constant in the chosen sector and
is added once during measurement.

Core source functions are in src/bosonic/kagome/simple_update_u1_cells.jl
and src/bosonic/kagome/native_measurement/. The native CTMRG retains separate
Q/R factors, handles charged transverse exchange explicitly, and saves
energy, local magnetization, bond and triangle observables plus convergence.
It supports ordinary3x3 and tied sqrt3xsqrt3 cells through the existing geometry.
The earlier Python measurement bridge has been removed.

For the long D4..12 job, see SU2_D4_D12_SERVER.md and run_su2_D4_D12_server.jl.
CPU dependencies use Project.toml; GPU dependencies use environments/gpu/Project.toml.
The main Julia entry selects and instantiates the appropriate Julia project.

## Local D4 numerical check, 2026-10-29

The bundled evolved SU2 seed was projected with component=1 and continued for
600 U1 steps with the short default schedule above. All bonds remained D=4.
Energies below include h=0.38 and are per physical spin.

| Environment | Projected initial E | U1 evolved E | Final m | CTMRG convergence |
| --- | --- | --- | --- | --- |
| chi=16 | -0.4336944270432336 | -0.43741960916094624 | 0.1111110920490275 | both converged: 10 / 7 iterations |
| chi=24 | -0.4336944396430512 | -0.4374196134316011 | 0.11111111057450687 | initial 50 iterations, error 2.49219e-6; final converged in 7 |

The final chi=16/24 energies differ by 4.27e-9 per spin. The energy reduction
is about 0.00372518 per spin. This is a short relaxation, not a claim of full
SU convergence or a proof of plateau stability. Fixed auxiliary charge keeps
the state in the selected 1/9 magnetization sector.

A numerical audit passed 90 checks: saved projection coefficients, unchanged
source file, U1 tensor types, D<=4, auxiliary-charge conservation, checkpoint
stage counts, exchange/field energy accounting, and site/bond/triangle outputs.
A representative SU weight initially had a triplet degeneracy
[0.2331989324,0.2331989324,0.2331989324]; after U1 evolution it split into
[0.1778053518,0.1171912565,0.2844086893]. These are simple-update weights,
not an exact many-body entanglement spectrum.

Local artifacts: results/u1_from_su2_D4_20261029_080336/.
The separate chi=16 remeasurement is recorded in chi16_recheck.toml and the
initial_ctm_chi16/ and final_ctm_chi16/ subdirectories. The saved initial/final
JLD2 files can be remeasured without repeating SU.

The D4 two-step regression in tests/test_u1_from_su2.jl passed 41/41 checks
with TensorKit 0.17.0 and 41/41 with TensorKit 0.14.11 (Julia 1.11.6).
It independently contracts the nine-spin torus to check total Sz=-1/2,
m=1/9 and zero Sz variance, verifies checkpoint wavefunction fidelity,
and checks that the original seed and input tensors are unchanged.
This finite-torus oracle is used only in this small regression test.
The trial workflow's energy measurements use infinite-system CTMRG.
