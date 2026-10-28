# Local validation, 2026-10-28

All runs used Julia 1.11.6, CPU, and 10 BLAS threads. TensorKit versions
were tested in separate Julia processes/environments; no default environment
or original numerical source file was edited.

| Check | TensorKit 0.14.11 | TensorKit 0.17.0 |
|---|---:|---:|
| Periodic geometry, including quotient partitions | 414 / 414 | 414 / 414 |
| 1x1 vs original itebd, real and complex tensors | 46 / 46 | 46 / 46 |
| Uniform lift onto both larger cells | 134 / 134 | 134 / 134 |
| Nonuniform sqrt3 vs expanded 3x3, real ansatz export | 182 / 182 | 182 / 182 |
| Explicit cross-cell down-triangle update | 9 / 9 | 9 / 9 |
| Checkpoint continuation, finite energies, input checks | 280 / 280 | 280 / 280 |
| Launcher / stage resume / overwrite prevention | 150 / 150 | 150 / 150 |
| **Total per version** | **1215 / 1215** | **1215 / 1215** |

`runtests.jl` contributes 1065 assertions; `workflow_smoke.jl` contributes
150. The export test constructs both the original `Kagome_iPESS` and
`Kagome_iPESS_immutable`, calls the original `build_A_kagome`, and checks
that physical fusion preserves the tensor norm. No substitute ansatz is
used for that test.

The workflow test runs two single-step stages for each large cell, then
resumes from stage 1 and compares every physical tensor and all six lambdas
with the uninterrupted result. These are small correctness checks, not
production convergence calculations.

## Direct old/new numerical comparison

`compatibility_probe.jl` fills the reduced SU(2) blocks deterministically,
independently of random-constructor ordering, before applying identical
evolution in both versions. Both cells and Dmax = 3, 4, 6 were compared.

| Case | Largest triangle SU energy difference | Largest lambda spectrum difference |
|---|---:|---:|
| 3x3, Dmax=3 | 1.1102e-15 | 1.2212e-15 |
| 3x3, Dmax=4 | 1.1102e-15 | 1.2212e-15 |
| 3x3, Dmax=6 | 1.6653e-15 | 1.0547e-15 |
| sqrt3xsqrt3, Dmax=3 | 7.7716e-16 | 5.5511e-16 |
| sqrt3xsqrt3, Dmax=4 | 7.7716e-16 | 5.5511e-16 |
| sqrt3xsqrt3, Dmax=6 | 1.1102e-15 | 8.8818e-16 |

Actual retained dimensions agree in all cases: dimension 3 for the Dmax=3
and Dmax=4 probes, and dimension 6 for Dmax=6. This illustrates that Dmax is
a bound and does not prescribe SU(2) representation content.

The energies above use the local SU lambda environment. No CTMRG energy,
finite-field magnetization, platform width, or thermodynamic convergence
claim follows from these tests. Compatibility of JLD2 checkpoints across
TensorKit versions was not assumed; continuation was checked within each
version. TensorKit 0.15 and 0.16 are allowed by the project but were not
separately run in this validation.

To reproduce the old/new comparison with your two environments:

```sh
julia --project=/path/to/old_env compatibility_probe.jl old.toml
julia --project=/path/to/new_env compatibility_probe.jl new.toml
julia compare_compatibility.jl old.toml new.toml
```

The comparison checks all retained dimensions and all reported energies /
lambda spectra, with absolute tolerance 1e-11, and exits with a test failure
if any check disagrees. The local comparison passed 43 / 43 assertions.

Full small-workflow reproduction:

```sh
julia --project=. runtests.jl
julia --project=. workflow_smoke.jl /path/to/new_validation_output
```
