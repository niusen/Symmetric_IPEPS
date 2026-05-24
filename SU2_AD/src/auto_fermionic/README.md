# AutoFermionicPESS

This folder contains a graded/fermionic starting point for the spin-Hall
fermionic PESS workflow.

The design is deliberately layered:

- `src/spaces.jl` uses TensorKit `FermionParity` sectors instead of ordinary
  bosonic `Z2Irrep` sectors.
- `src/local_operators.jl` defines the spinful local Fock space as two
  fermionic modes, with basis `|0>, |up>, |dn>, |up dn>`, plus explicit
  physics-basis to TensorKit-sector-basis conversion helpers for one or more
  sites.
- `src/pess_state.jl` defines a graded triangular PESS tensor pair and the
  PESS-to-iPEPS contraction in the legacy leg order.
- `src/spin_hall_model.jl` reproduces the legacy
  `Triangle_Hofstadter_Hubbard_spinHall` coefficients and builds dense or
  TensorKit three-site triangle Hamiltonians/gates in the correct sector
  basis.
- `src/entanglement.jl` provides spin trace, spin-partition RDM, entanglement
  spectrum, and von Neumann entropy helpers.
- `src/graded_double_layer.jl` starts the no-explicit-swap CTMRG migration by
  constructing a fused double-layer tensor from `FermionParity` iPEPS tensors.
- `src/legacy_adapter.jl` loads the existing CTMRG and observable code so the
  new layer can reuse your current environment contraction machinery.

## Why this is not just ordinary Z2

TensorKit now documents `FermionParity` as a fermionic sector with the minus
sign for exchanging two odd sectors implemented through the R-symbol. That is
the graded behavior needed here; an ordinary `Z2Irrep` only enforces block
sparsity and does not by itself encode fermionic exchange signs.

The implementation follows the graded-Hilbert-space viewpoint used in recent
fermionic tensor-network work: fermionic signs belong in tensor permutation and
contraction rules, not in hand-written observable code.

## Minimal usage

```julia
using Pkg
Pkg.activate(@__DIR__)

using AutoFermionicPESS

setting = SpinHallEnergySetting(Lx=2, Ly=2, magnetic_cell=2)
pars = SpinHallParameters(t1=1.0, t2=1.0, mu=0.0, U=0.0, mx=0.0, B=0.0)
coeffs = spin_hall_coefficients(setting, pars)

pess = random_triangle_pess(2, 2)
A = pess_to_ipeps_tensor(pess)
G = triangle_gate_tensormap(coeffs, 1, 1, 0.01)
```

For the current optimized states and CTMRG environment:

```julia
using AutoFermionicPESS

load_legacy_spin_hall!()

# Build A_cell, AA_cell, CTM_cell with your existing workflow, then:
result = evaluate_legacy_spin_hall(parameters, A_cell, AA_cell, CTM_cell,
                                   ctm_setting, energy_setting)
```

## Next integration step

The old code still contains explicit `swap_gate` calls in the CTMRG/update
layer. The intended migration path is:

1. Use this folder for spin-resolved density matrices and graded local gates.
2. Replace old `Rep[Z2Irrep]` physical spaces by `physical_spinful_space()`.
3. Move contractions that currently call `swap_gate` to TensorKit fermionic
   `permute` / `@tensor` calls.
4. Keep the old observable tests as regression tests while the backend is
   replaced.

## Legacy gate comparison

After `Pkg.test()` passes, run the optional comparison script:

```julia
include("D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/src/auto_fermionic/scripts/compare_legacy_spin_hall_gate.jl")
```

This loads the old swap-gate spin-Hall triangle gate and compares it with the
new dense/graded construction. The script requires `Zygote`, because the legacy
files use `@ignore_derivatives`; keep that as an optional dependency rather
than part of the small core package.

## Saved-state checks

Verify the old swap-gate CTMRG energy of the saved spin-Hall variational state:

```julia
include("D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/src/auto_fermionic/scripts/evaluate_saved_spin_hall_state.jl")
```

Convert the same saved `Triangle_iPESS_immutable` state to the graded
`FermionicTrianglePESS` representation and check that the raw `B` and `T`
tensors are preserved exactly as dense arrays:

```julia
include("D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/src/auto_fermionic/scripts/convert_saved_spin_hall_state.jl")
```

Compare the first no-explicit-swap graded double layer against the legacy
`build_double_layer_swap` result:

```julia
include("D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/src/auto_fermionic/scripts/compare_saved_double_layer.jl")
```

At this stage this last comparison is a diagnostic, not a pass/fail test. It
shows the remaining convention map needed between the old saved swap-gate PESS
state and the new graded contraction convention before replacing CTMRG.

Current diagnostic status:

- The old saved `B` and `T` tensors can be converted losslessly as raw block
  data.
- For old saved states, the safer bridge is to first form the legacy iPEPS
  tensor `A = permute(Tm * Bm, (1,5,4,2,3))`, then convert that whole `A` to
  `FermionParity`.
- Converting `B` and `T` separately and then contracting them with graded
  operations gives a different state convention.
- The double-layer comparison is still not expected to match: direct dense
  array comparison between old `Z2Irrep` fusion trees and `FermionParity`
  fusion trees is not a categorical equality test, and simple external parity
  gauges or the legacy fused-leg parity corrections do not repair the mismatch.

The next real migration step is therefore to write the CTMRG double layer as a
graded diagram directly, using planar reshaping for bookkeeping and explicit
TensorKit braiding only where the diagram has crossings, instead of translating
the old swap-gate implementation line-by-line.

The package currently exposes two double-layer builders:

- `graded_build_double_layer(A)`: a line-by-line migration scaffold based on
  the old fusion path. It is useful for diagnostics but is not the final CTMRG
  replacement.
- `graded_build_double_layer_direct(A)`: a more diagrammatic builder that
  traces the physical leg of `A'` and `A` directly, then fuses the four virtual
  bra/ket pairs. This is the preferred starting point for a native graded CTMRG
  path.

The first branch-A CTM skeleton is now available:

- `graded_double_layer_cell(A_cell; builder=:direct)` builds a cell of graded
  double-layer tensors.
- `graded_init_ctm(AA, U_L, U_D, U_R, U_U; init_type="PBC")` initializes the
  four corners and four edge tensors for one site.
- `graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)` runs both
  steps on a unit cell.
- `graded_ctm_projectors(CTM, AA)` constructs the left-move CTMRG projectors
  from a native graded environment.
- `graded_ctm_left_update(CTM, AA)` performs one no-explicit-swap left
  absorption step for a single site.

For the saved spin-Hall state, run:

```julia
include("D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/src/auto_fermionic/scripts/graded_skeleton_saved_state.jl")
```

This only initializes the graded double-layer and CTM tensors. It deliberately
does not run CTMRG iterations or compute energy yet.

Run one native graded left absorption step on the same saved state:

```julia
include("D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/src/auto_fermionic/scripts/graded_left_update_saved_state.jl")
```

This still does not compute energy. It is the first tested CTMRG update piece
for branch A.

The branch-A unit-cell update now has a first directional version:

- `graded_ctm_cell_directional_update(CTM_cell, AA_cell; direction, chi)`
  applies one periodic unit-cell absorption step.
- `graded_ctm_cell_sweep(CTM_cell, AA_cell)` applies the default direction
  order `(3, 4, 1, 2)`.
- `graded_ctm_cell_iterate(CTM_cell, AA_cell; maxiter, chi)` repeats sweeps
  and records a simple norm-signature history for early convergence checks.

For the saved 2x2 spin-Hall state, run a single direction update:

```julia
include("D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/src/auto_fermionic/scripts/graded_cell_update_saved_state.jl")
```

Run one full branch-A CTMRG sweep and print the spectrum-based convergence
monitor:

```julia
include("D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/src/auto_fermionic/scripts/graded_iterate_saved_state.jl")
```

That script also prints the first native graded onsite observables
`n_total`, `n_double`, and `Sx/Sy/Sz` for each site in the 2x2 cell.

The native observable layer currently includes:

- `graded_ob_onsite_at(...; site=:LU|:RU|:LD|:RD)`, which inserts one even
  onsite operator at any corner of the 2x2 environment patch.
- `graded_ob_onsite_cell`, which evaluates one onsite operator over the unit
  cell.
- `graded_ob_product_2x2`, which inserts products of even onsite operators on
  selected 2x2 patch corners, e.g. `(LU=ops.n_total, RD=ops.n_double)`.
- `graded_ob_triangle_2x2(...; orientation=:up|:down)`, which inserts a
  genuine three-site even TensorMap into a 2x2 patch. The `:up` convention is
  `(LD, RD, RU)`, and the `:down` convention is `(LD, LU, RU)`, matching the
  legacy triangle comments.

For multi-site operators the normalization uses the same multi-site diagram
with the categorical identity `TensorKit.id(codomain(O))`. This matters in the
fermionic graded setting: a plain dense identity matrix is not the same object
as the categorical identity on a product of graded spaces.

Odd hopping terms are intentionally not routed through the even-product
interface. They are handled as part of a genuine multi-site even operator, such
as `triangle_hamiltonian_tensormap`.

## References checked

- TensorKit manual, "Fermionic sectors": `FermionParity` has fermionic
  braiding and the odd-odd exchange sign through the R-symbol.
- PEPSKit manual: PEPSKit lists native support for TensorKit symmetric tensors,
  including fermionic tensors.
- Mortier et al., SciPost Phys. 18, 012 (2025): fermionic tensor networks via
  graded Hilbert spaces avoid Jordan-Wigner strings and explicit swap-gate
  tracking on arbitrary graphs.
