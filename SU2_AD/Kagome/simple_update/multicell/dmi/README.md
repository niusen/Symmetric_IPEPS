# Kagome Dz DMI (pure Julia TensorKit)

Functional files are in `src/bosonic/kagome/dmi/`. They reuse the existing
`KagomeU1Cells` SU/Trotter/HOSVD kernels and `KagomeNativeMeasurement` factorized,
memory-bounded CTMRG and operator insertions. No Python program is called.
Old Heisenberg files are unchanged.

## Directory layout

```text
dmi/
  server/       editable run options and background .sh launchers
  experiments/  pure-DMI order probe and historical candidate checks
  tests/        small-D correctness checks and independent-seed smoke test
  data/logs/    local and server logs/PID records
  data/tests/   small-test results and the preserved pre-reorganization baseline
  README.md
```

Algorithm functions remain in `SU2_AD/src/bosonic/kagome/dmi/` and reuse the
existing kagome SU/CTMRG functions. Source modules are not copied into this run
folder. Checkpoints and scientific results remain under multicell `results/`;
the saved D4 seed and Project.toml/Manifest.toml remain under multicell.

The current unrestricted random scan is configured in
[server/run_independent_random_D2_D4.jl](server/run_independent_random_D2_D4.jl).
From **dmi/server/** run `sh run_independent_random_D2_D4.sh`.
It uses D=2/3/4, Dz=.02/.05/.1, seven fields, and three independent random
seeds at every point (189 optimizations), CPU/10 threads, chi=12, CTMRG max50,
tol=1e-6. All physics settings remain in the .jl file. Background logs and PID
records go to **dmi/data/logs/**. Old PID records do not identify live tasks.
Re-submit unchanged settings to resume the same output directory. No watchdog.

Upload the current `dmi/` directory excluding `data/`, together with the
updated `src/` dependencies. Keep the directory depth unchanged; a multicell
parent renamed `u1_multicell` still works. The old flat .jl/.sh paths have moved.

For a small correctness test, from the multicell directory run:

```sh
julia --project=. dmi/tests/runtests.jl
julia --project=. dmi/tests/test_independent_random_fields.jl
```

These use small tensors and temporary output directories. The longer
**experiments/** scripts are separate from the small correctness tests.

## Single-run options

Edit options in `Kagome/simple_update/multicell/dmi/server/run.jl`, then run that file
with Julia or use `sh run.sh` in the same directory (background log/PID).
CPU and Dz=0 are defaults. Production mode defaults to U1, using the saved D4
plateau seed; adjust the input path to your saved state. All options are in code.
The existing multicell Project.toml/Manifest.toml are reused. The default input
is `data/u1_D4_optimized_seed.toml`, the existing version-neutral D4 optimized
seed including lambdas. Raw old server JLD2 files may require their original
TensorKit version; a bare `u1_state.toml` export lacks SU weights and is not
silently substituted. New U1 checkpoints include `*.seed.toml` companions with
all tensors, weights, charge order and DMI settings; the reader prefers these
when available. Same-version dense JLD2 checkpoints are supported directly.

For an unrestricted pure-DMI test choose `representation=:dense`, `input=""`,
`J=0., Dz=1., h=0.` and a fresh output path. Random dense states have no auxiliary
charge, so transverse moments can develop. Converting a fixed-sector U1 seed
to dense retains that ket/sector; it is not an unrestricted random initialization.

The Hamiltonian is `J sum SS + Dz sum_CCW kappa_z + h sum Sz` (legacy Julia **+h**).
`Dz` defaults to zero; `J=0` is allowed. U1 remains valid at Dz≠0, SU2 does not.
Both up and down triangle simplex axes have the original **clockwise** physical
order, so the cyclic DMI operator has a minus sign to implement CCW arrows.
Each triangle gate receives half the field, since each site is on two triangles.

## Outputs and limitations

The run writes `initial/latest/final.jld2`, `observables.toml/.json/.jld2`,
`site_magnetization.csv` and `bond_energies.csv`. Bonds include SS, CCW two-spin
vector chirality and JSS+Dz*kappa energy. The field is counted once per site.
Intermediate non-Hermitian SpSm expectations remain complex until the complete
Hermitian observable is formed. This is essential for nonzero DMI energy.

Every measurement saves CTMRG convergence, timing and the maximum imaginary
residual of Hermitian observables. `measurement_reliable` requires a converged
CTM and imaginary residual below 1e-8. A deliberately underconverged small test
can produce larger residuals and is not a quantitative ground-state result.
The DMI checkpoint has a separate format to prevent old Heisenberg-only
measurement/variance programs from silently ignoring Dz. Partial checkpoints
can explicitly seed a new run, which repeats the configured schedule.

U1 evolution keeps the seed's auxiliary charge sector. A 1/9 seed at h=0
does not search all magnetizations; U1 single-site Sx,Sy are exactly zero.
The dense test path can measure Sx,Sy directly. U1 magnetic order instead
requires appropriate transverse long-distance correlations.

## Tests

Run `dmi/tests/runtests.jl` with the multicell Julia project. The checks use
dense D1/D2 tensors and a small projected U1 fixture (actual D≤3). They include
Hermiticity, total-Sz conservation, Dz=0 regression, field counting, both triangle
directions, short complex SU, U1/dense nine-spin wavefunctions and output files.

At J=h=0, the coherent q=0 120-degree product reference has energy per spin
`-sqrt(3)*Dz/4`. This is a sign/geometry check, not the exact quantum energy.
Python uses the opposite field parameter sign (-h Sz); reverse h to compare
the same ket. The DMI convention is the same.
