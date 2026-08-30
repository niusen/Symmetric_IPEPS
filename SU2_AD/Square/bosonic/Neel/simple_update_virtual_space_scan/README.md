# 2×2 SU(2) Simple-Update virtual-space scan

This directory is intentionally self-contained at the experiment level.  It
uses the repository's square-lattice Simple Update and CTMRG implementation,
but does not modify source files while scanning initial virtual spaces.

Reference target for the square-lattice spin-1/2 Heisenberg model (`J=1`):

- SU(2) iPEPS, `D*=4`, total `D=12`: `e0 = -0.6686` per site
- QMC: approximately `-0.6694` per site

Initializations include homogeneous integer/half-integer direct sums and the
paper spaces

```text
Veven = {0^1, 1^2, 2^1}          D*=4, D=12
Vodd  = {(1/2)^2, (3/2)^2}       D*=4, D=12
```

placed on a 2×2 perfect matching.  Columnar and ABBA-like staggered choices
are both tested.

Run a quick screen with:

```bash
julia screen.jl
```

Environment variables control seeds, schedules and CTM accuracy; see the top
of `screen.jl`.  Every case is saved below `results/<run-id>/<case>/`, and a
CSV summary is written to the run directory.

By default only `initial.jld2` and `final.jld2` are saved.  Intermediate
stage states are unnecessary for reproduction and can be large because they
contain the per-sweep history.  Set `SCAN_SAVE_STAGES=true` only when an
intermediate restart point is explicitly needed.

Resume a saved state with a finer schedule:

```bash
SCAN_CONT_SCHEDULE=fine_long julia continue_saved.jl results/<case>/final.jld2 <output-name>
```

Generate per-stage local energies and sector-resolved lambda values:

```bash
julia analyze_saved_series.jl results/<case>
```

The completed scan and selected state are documented in `RESULTS.md`.

## Continue with square-lattice Full Update

`Full_update_J1_SU2_cell.jl` now accepts the Simple Update `final.jld2`
directly through its `T_set` key.  The lambda tensors are not absorbed: the
saved `T_set` is already the physical iPEPS state.  From
`Square/bosonic/Neel` run, for example,

```bash
FU_INIT=simple_update_virtual_space_scan/results/paper_y_staggered_seed666_fine_long/final.jld2 \
FU_LX=2 FU_LY=2 FU_DMAX=12 FU_MULTIPLET_TOL=1e-5 \
FU_CHI=32 FU_TAU=0.01 FU_DT=0.01 \
julia Full_update_J1_SU2_cell.jl
```

For every x or y bond the two rank-5 tensors are first split into fixed
rank-4 outer tensors and two rank-3 bond tensors.  The gate and
`truncdim(Dmax; multiplet_tol=...)` act only on the reduced bond tensor; the
2×1 or 1×2 CTM cluster then optimizes the two rank-3 tensors alternately.
As in the old triangular-lattice Full Update, CTM is reconstructed from
scratch after every local bond update; the previous CTM is not reused even
when the multiplet spaces remain unchanged.  The cell FU therefore requires
`FU_REFRESH_CTM=true` (the default) and rejects `false`.

The lightweight structural checks are `probe_fu_reduced.jl` (SVD
reconstruction, gate/truncation, and cross-layer spaces) and
`probe_fu_ctm.jl` (one x/y bond in a small-D CTM environment).  Set
`FU_PROBE_D12=true` when running the first probe to check the saved D=12
state; its default is the faster total-D=3 test.
