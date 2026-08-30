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
