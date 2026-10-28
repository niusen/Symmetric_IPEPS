# Kagome SU(2) / U(1) Simple Update on larger cells

U(1) conversion and evolution are now available: see [U1_USAGE.md](U1_USAGE.md).
To edit parameters directly in Julia, open **`examples/run_kagome_settings.jl`**,
change the settings at the top, and run the whole file. No arguments are
needed; the default task is a six-step U(1) polarized trial.
For the longer SU(2) D=2,3,4 convergence study, use **examples/run_su2_D234.jl**;
results and limitations are recorded in [SU2_D234_RESULTS.md](SU2_D234_RESULTS.md).
The spatial cell diagram, plotting script and virtual-space snapshot are in
[visualization/](../visualization/README.md).
Conversion defaults to U(1), with an explicit dense export option. The SU
runner retains its SU(2) default; `--symmetry u1` selects U(1). An existing
charged SU(2) state can initialize U(1) through exact component projection.

This adds rectangular and sqrt(3) x sqrt(3) cells to the existing Julia
kagome Simple Update. All implementation and example files are new.

| Cell argument | Independent PESS | Physical spins | Exported CTMRG grid |
|---|---:|---:|---|
| `1x1` | 1 | 3 | 1 x 1 |
| `sqrt3xsqrt3` | 3 | 9 | 3 x 3, with tied representatives |
| `3x3` | 9 | 27 | 3 x 3 |

The library also accepts arbitrary positive rectangular `Lx,Ly`.

## Tensor and lattice conventions

The conventions are exactly those of `src/bosonic/kagome_AD_SU2_cell.jl`
and `src/bosonic/kagome_model_cell.jl`. Coordinates have x to the right and
y down. B tensor axes remain **(down, up, physical)**. Each cell stores
`T_u,T_d,B_a,B_b,B_c` and six separate lambda tensors.

The original simplex virtual-axis order is:

```text
T_u(x,y): B1(x,y),   B2(x,y), B3(x,y)
T_d(x,y): B1(x+1,y), B2(x,y), B3(x,y+1)
```

Both lambda tensors of a physical B tensor are stored at that B's coordinate.
Thus a down update gathers tensors AND their up lambdas from three cells,
then returns each updated B and down lambda to its original owner.

For sqrt3 cells, `color(x,y) = 1 + mod(x-y,3)`. The periodic translations
are `(1,1)` and `(-1,2)`. There are exactly three updates per triangle layer,
not nine updates of tied copies. Arbitrary independent tensors are allowed
in each representative, including unequal virtual SU(2) spaces after SVD.

The local `Tri_T_up`, `Tri_T_dn`, HOSVD and multiplet truncation are included
from the original files, unchanged. The update sequence is also unchanged:

```text
down half-step, (up full-step, down full-step)*round(tau/dt), up half-step
```

Rotation-symmetric HOSVD is disabled because inequivalent bonds must remain
independent. No cell averaging, zero padding, or extra projection is applied.
`Dmax` counts total states, including the `(2j+1)` SU(2) degeneracy; it is
not the number of multiplets and does not prescribe their representation
content. The actual dimension can be smaller than `Dmax`.

## Environment and old/new TensorKit support

Dependencies: TensorKit, ChainRulesCore and JLD2. The supplied `Project.toml`
accepts TensorKit 0.14 through 0.17; no global Julia environment is modified.

The local `simple_update_compat.jl` adapter follows the approach used in
your DMRG `src/settings.jl`: it chooses existing legacy `tsvd/eigh` or newer
`svd_trunc/eigh_full` functions by API availability. It also adapts
`truncdim/truncerr` and `permute` calling conventions. These bindings are
confined to the new module, without adding methods to TensorKit itself.
The original multiplet-boundary removal and truncation tolerances remain
in effect. See `VALIDATION.md` for versions actually tested.

From this directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. tests/runtests.jl
```

An existing compatible environment can also be used directly:

```sh
julia --project=/path/to/your/environment tests/runtests.jl
```

## Run

Small workflow checks (two full steps plus original boundary half-steps):

```sh
julia --project=. ../../../src/bosonic/kagome/workflows/run.jl --cell sqrt3xsqrt3 --Dmax 3 --smoke
julia --project=. ../../../src/bosonic/kagome/workflows/run.jl --cell 3x3 --Dmax 3 --smoke
```

Longer evolution:

```sh
julia --project=. ../../../src/bosonic/kagome/workflows/run.jl --cell sqrt3xsqrt3 --Dmax 6 --seed 1234 --out results/sqrt3_D6
julia --project=. ../../../src/bosonic/kagome/workflows/run.jl --cell 3x3 --Dmax 6 --seed 1234 --out results/3x3_D6
```

The default schedule uses the user-requested values:

| dt | tau | round(tau/dt) |
|---:|---:|---:|
| 0.03 | 40 | 1333 |
| 0.01 | 20 | 2000 |
| 0.003 | 5 | 1667 |
| 0.001 | 2 | 2000 |
| 0.0005 | 1 | 2000 |

Total: 9000 complete up/down pairs per new point, plus boundary half-steps.
There is no automatic extension or energy convergence stopping. Ten BLAS
threads are used by default; `--threads N` changes this.

The initial virtual space defaults to `0 + 1/2`, dimension 3. To choose its
representation content explicitly, for example:

```sh
julia --project=. ../../../src/bosonic/kagome/workflows/run.jl --cell 3x3 --Dmax 6 --virtual '0:1,1/2:1,1:1'
```

`--virtual` entries are `spin:multiplicity`. The SVD can change the selected
space during evolution. Pure `1/2` on every B leg admits no nonzero B tensor
in this singlet construction and is rejected during initialization.

Every completed stage saves its physical tensors, all six lambda tensors,
Hamiltonians, cell mapping, and metadata in `stage_XX.jld2`. The final state
is saved in `final.jld2`; actual bond dimensions and SU(2) spaces are written
to `su_diagnostics.txt`. New runs require a new output directory.

To continue from the last completed stage after interruption:

```sh
julia --project=. ../../../src/bosonic/kagome/workflows/run.jl --cell 3x3 --Dmax 6 --resume results/3x3_D6/stage_02.jld2 --out results/3x3_D6_continued
```

Keep the same cell, Dmax, J1 and schedule; use `--smoke` again if the source
was a smoke test. An interrupted stage is repeated from the preceding saved
stage. Loading checkpoints is tested within each TensorKit version; changing
TensorKit versions while loading JLD2 tensor objects is not assumed safe.

## Use from existing CTMRG / optimization code

```julia
include("../../../src/bosonic/kagome/simple_update_cells.jl")
using .KagomeSU2Cells

g = KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3) # or :rectangular
state = initialize_cells(g, SU2Space(0=>1,1/2=>1);rng=MersenneTwister(1234))
H = heisenberg_triangle()
state = itebd_cells(state,H,0.1,0.01;Dmax=6)

# Kagome_iPESS must be loaded from the existing ansatz code by the caller.
x = to_ipess_cells(state,Kagome_iPESS)
```

The export follows the original SU physical-tensor convention: no extra
lambda factors are absorbed. Exported sqrt3 grid entries are separate deep
copies; a subsequent optimizer must enforce `state.geometry.site_map` if
it is intended to preserve sqrt3 parameter tying. The helper does not add
new-version compatibility to the repository's existing CTMRG/AD files.

The default SU(2) runner performs zero-field singlet Simple Update and does
not launch CTMRG. Its lambda-environment energy estimates are SU diagnostics,
not CTMRG or variational energies. The U(1) option supports charged input and
separate block U(1) CTMRG measurements, as described in U1_USAGE.md.

## Pure Julia measurement and D4..12 server task

The active conversion/measurement path is entirely Julia/TensorKit. See
SU2_D4_D12_SERVER.md and run_su2_D4_D12_server.jl. Native functions are in
src/bosonic/kagome/native_measurement/; no Python bridge is required.
