# U(1) validation, 2026-10-28

Environment: Julia 1.11.6, TensorKit 0.14.11 and 0.17.0, 10 BLAS threads.
No production field scan or long-time SU calculation was run locally.

## Exact conversion and SU tests

`test_u1_cells.jl`: **167 checks passed in each TensorKit version**:

| Group | Checks per version |
|---|---:|
| Exact restriction, irrep multiplicities, duals, real/complex coefficients, auxiliary component projections | 132 |
| U(1) SU at Dmax=2,3, conserved Sz, checkpoint continuation, independent charge-state truncation | 19 |
| Default U(1) export, explicit dense fallback, U(1) re-export | 7 |
| Both cell geometries, charged U(1) and existing SU(2) trial entry points | 9 |

`test_u1_initialization.jl`: **51 additional checks passed in each version**.
It exercises charged SU(2) and U(1) `--init` paths, including expansion
from three tied sqrt3 PESS to nine independent PESS in a 3x3 cell. Starting
from the same copied state, two full SU steps in the two geometries agree
tensor by tensor to tolerance 3e-11; the 27-spin cell has three auxiliary
charges and the same normalized 1/9 magnetization sector. The sqrt3
magnetization is also verified by an explicit finite wavefunction contraction.
Total: **218 checks per TensorKit version**, 436 across both versions.

Restriction tests compare coefficients in the correct charge-ordered basis
and compare the complete normalized nine-spin wavefunction for auxiliary
spin 1/2 and 3/2 and every component. They also compare energy, site
magnetization and S^2 with the original SU(2) contraction. Conversion is
exact to rounding; it does not change the physical projected state.

The U(1) truncation test explicitly cuts a tied pair of singular values in
different charge sectors to dimension one. It does not demand an asymmetric
set of bond charge labels, since the charge origin of an SVD bond depends
on gauge and the initialization may preserve extra symmetries.

The Dmax=3 charged trial starts from a random SU(2) 0 plus 1/2 initialization
with seed 1234 and auxiliary j=1/2, component 1. It uses dt:tau pairs
0.01:0.02, 0.003:0.006, 0.001:0.002 (six complete steps), measuring the
finite nine-spin torus at h=0.38:

| Quantity | Projected initial state | After U(1) SU |
|---|---:|---:|
| E(h)/spin | -0.343410017178144 | -0.345059213264099 |
| normalized magnetization | 0.111111111111111 | 0.111111111111111 |
| S^2 | 0.750000000000000 | 0.750633363637348 |

These are workflow checks, not converged ground-state energies or a
SU(2)-versus-U(1) accuracy benchmark. The S^2 change confirms that U(1)
evolution can leave the SU(2)-restricted variational family while retaining
the selected Sz sector.

## Infinite-lattice U(1) CTMRG versus independent dense path

Python 3.12.8, PyTorch 2.5.1, local RTX 4050 (`cuda:0`), complex128,
10 CPU threads. Server commands retain `cuda:1` as the default.

Input: the existing six-step SU(2) D=3 checkpoint with SHA256
`b0c38bbc9735c2ae52be4cafa655e51222947fcf58654afc0bf0fa2d0ce9cfb2`,
projected to auxiliary m=+1/2. This comparison uses the same physical
state in both representations, before any additional U(1) evolution.

At h=0.38, chi=16, tolerance=1e-6:

| Quantity | Block U(1) | Dense |
|---|---:|---:|
| E/spin | -0.35507234547019095 | -0.35507234547020000 |
| normalized magnetization | 0.11111110918674216 | 0.11111110918671614 |
| CTMRG iterations | 18 | 18 |
| final corner-spectrum error | 5.273036082435978e-7 | 5.701589771e-7 |

Absolute differences: E=9.05e-15, exchange energy=1.39e-14,
field energy=4.94e-15, magnetization=2.60e-14. The field-energy
reconstruction check passed. Actual U(1) corner dimensions ranged 13..16.
Both calculations converged within the specified environment tolerance.
The U(1) measurement used genuine YASTN block tensors through double layers,
CTM moves and observables; it did not run the dense measurement backend.

This checks representation consistency and operator/arrow conventions,
not chi convergence or speed. Small-D GPU block overhead can be significant.
The input is a short trial, and exact m=1/9 is imposed by the selected charge
sector; no plateau width is inferred from these tests.
