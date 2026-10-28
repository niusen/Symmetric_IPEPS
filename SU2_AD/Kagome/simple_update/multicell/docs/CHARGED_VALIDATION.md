# Auxiliary-spin prototype validation, 2026-10-28

Julia 1.11.6, CPU, 10 BLAS threads. All implementation and test files are
new; the existing kagome SU and multicell source files were not changed.

## Correctness checks

`tests/test_charged_cell.jl` passes **86 / 86 checks** in each of TensorKit
**0.14.11** and **0.17.0**:

| Check | Assertions per version |
|---|---:|
| j=1/2 and j=3/2, all three auxiliary-leg placements | 56 |
| Trivial j=0 spectator recovers the original up update | 4 |
| Untruncated charged HOSVD matches physical gate action | 2 |
| Magnetization, dense projection, checkpoints and continuation | 24 |

The first group verifies physical total Sz, S^2, normalized magnetization,
Zeeman-energy shifts and the degeneracy of opposite m components on the
finite nine-site torus. The last group also checks that tracing the extra
index gives zero magnetization, while explicit polarization and subsequent
dense contraction reproduce the SU(2)-network wavefunction.

## Six-step trial

Settings: sqrt3xsqrt3, one auxiliary spin j=1/2 on Tu of representative 1,
highest-weight auxiliary component, random seed 1234, initial virtual
space `0 + 1/2`. The schedule contains only:

```text
dt=0.01,  tau=0.02   (2 complete steps)
dt=0.003, tau=0.006  (2 complete steps)
dt=0.001, tau=0.002  (2 complete steps)
```

Original boundary half-steps are included at every stage. All energies
below are **per physical spin on a finite nine-site periodic torus**,
not infinite-system / CTMRG energies and not converged variational values.

| Dmax | Initial exchange energy | Final exchange energy | Final energy at h=0.38 | Normalized magnetization |
|---:|---:|---:|---:|---:|
| 3 | -0.322298906067033 | -0.322298906067033 | -0.343410017178144 | 0.111111111111111 |
| 6 | -0.322298906067033 | -0.330648572019883 | -0.351759683130994 | 0.111111111111111 |

Both trials retain physical total Sz=-1/2 and S^2=3/4 within numerical
precision. The exact lowest exchange energy in this finite torus's
Sz=-1/2 sector, obtained from a 126-dimensional matrix, is
`-0.4410001944365022` per spin. The trial energies respect this lower bound
and remain far from it, as expected for such short runs.

The Dmax=3 trial has no resolved energy improvement during this schedule.
Increasing the bound to Dmax=6 gives an energy decrease of approximately
0.00834967 per spin. This is a functional demonstration, not a claim that
all initializations or subsequent SU steps lower the projected energy.

Old/new TensorKit trials agree to approximately 1e-15 in exchange energies,
site Sz and normalized magnetization. Each trial saves initial, per-stage
and final raw SU(2) states, polarized dense tensors, and a TOML report.

## Interpretation

The construction successfully realizes the proposed SU(2)-covariant
magnetized PESS. **m=1/9 is selected by the auxiliary boundary condition**;
its persistence under evolution is a sector-conservation check, not an
independent discovery of a magnetization plateau. Establishing plateau
stability requires infinite-system measurements and competition with other
magnetization sectors. The SU HOSVD's multiplet-summed environment is also
an approximation to the polarized-state optimization problem, as detailed
in `CHARGED_CELL.md`.
