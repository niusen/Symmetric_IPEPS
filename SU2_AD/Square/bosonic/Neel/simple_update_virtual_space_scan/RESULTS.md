# Square J1 SU(2) Simple-Update scan results

## Outcome

The best tested virtual-space structure is the paper's `D*=4`, `D=12`
assignment with three even legs and one odd leg at every tensor:

```text
Veven = {0^1, 1^2, 2^1}          D*=4, D=12
Vodd  = {(1/2)^2, (3/2)^2}       D*=4, D=12
```

For the 2x2 cell, the best tested matching (`paper_y_staggered`) is:

```text
x(1,1), x(2,1), x(1,2), x(2,2): Veven
y(1,1) -> (1,2):                 Vodd
y(1,2) -> (1,1):                 Veven
y(2,1) -> (2,2):                 Veven
y(2,2) -> (2,1):                 Vodd
```

Thus the two odd bonds form a staggered perfect matching and every site is
incident on exactly one odd bond.  The selected saved state is:

```text
results/paper_y_staggered_seed666_fine_long/final.jld2
```

Its best directly measured energy is `E/site = -0.6545092539992814` at
`chi=32` (CTMRG error `5.05e-7`).  The paper reference is `-0.6686` at
`chi*=128`; the remaining difference is about `0.01409`.  These are not
equal-chi calculations, and the paper state is variationally optimized,
whereas this scan deliberately uses Simple Update only.

## Initial-space screen

All rows below use seed 666 and the legacy schedule
`30@0.1, 20@0.05, 20@0.01, 4@0.002`.  These are the inexpensive local
Simple-Update environment energies used only for screening.

| initialization | SU E/site | final qualitative structure |
|---|---:|---|
| paper_y_staggered | -0.6012939107 | 6 paper-even + 2 paper-odd |
| mixed_broad | -0.6002682183 | 2 paper-even + 6 paper-odd |
| paper_x_staggered | -0.5994352059 | 2 odd; one even bond changed to `{0^1,1^3}` |
| paper_y_columnar | -0.5980799509 | paper matching with one changed even bond |
| paper_x_columnar | -0.5979065131 | 2 paper-odd + 6 paper-even |
| mixed_balanced | -0.5842549590 | lower-D even/odd spaces |
| mixed_min | -0.5842418231 | lower-D even/odd spaces |

The exact paper union, initially
`{0^1,(1/2)^2,1^2,(3/2)^2,2^1}` (`D*=8`, `D=24`), was also tested.  It
first truncated to a mixed `D*=4`, `D=8` space on every bond.  Long evolution
then spontaneously produced the correct 6-even/2-odd parity matching, but
selected the inferior spaces `{0^2,1^2}` (`D=8`) and
`{(1/2)^3,(3/2)^1}` (`D=10`), with SU energy `-0.5796774266`.

This shows that a mixed even+odd initialization can discover the parity
pattern, but does not reliably discover the paper's multiplet content.  Direct
initialization with `Veven` and `Vodd` is more reliable for this Simple-Update
workflow.

## Time-step convergence

After the legacy schedule, the two leading sectors were continued with
`10@0.002, 6@0.001, 3@0.0005, 1@0.0002`.  No early stopping was used.

| state | dt=0.002 | dt=0.001 | dt=0.0005 | dt=0.0002 |
|---|---:|---:|---:|---:|
| paper_y_staggered, seed 666 | -0.6012940112 | -0.6014520997 | -0.6015311125 | -0.6015785097 |
| paper_y_staggered, seed 1234 | -0.6013214105 | -0.6014814045 | -0.6015613651 | -0.6016093305 |
| mixed_broad, seed 666 | -0.6002682933 | -0.6004343398 | -0.6005173409 | -0.6005671337 |

All three sequences improve monotonically as `dt` decreases.  Their final
lambda spectra and bond spaces are in each result directory's
`lambda_report.txt`.

## CTMRG comparison

The CTMRG measurement uses 2x1 and 1x2 two-site density matrices, not a 2x2
measurement cluster.

| state | chi=8 | chi=16 | chi=24 | chi=32 |
|---|---:|---:|---:|---:|
| paper_y_staggered, seed 666 | -- | -0.6492320387 | -0.6518355416 | **-0.6545092540** |
| mixed_broad, seed 666 | -0.6265265782 | -0.6501437493 | -0.6516295376 | -0.6533485568 |
| paper_y_staggered, seed 1234 | -- | -- | -0.6515016889 | -- |

At low environment dimension (`chi=8,16`) the opposite 3-odd/1-even sector
looks better.  The ranking crosses by `chi=24`; at `chi=32` the paper
3-even/1-odd structure is lower by about `0.00116`.  Therefore low-chi CTMRG
alone would select the wrong virtual parity sector in this scan.

Four seeds were screened for `paper_y_staggered`.  Their legacy-schedule SU
energies were seed 1234: `-0.60132108`, seed 666: `-0.60129391`, seed 2021:
`-0.59552437`, and seed 42: `-0.59317295`.  Although seed 1234 had a slightly
better local SU energy, its `chi=24` CTMRG energy was worse than seed 666, so
seed 666 remains the selected state.

## Reproduction files

- `screen.jl`: initialization/seed/schedule screen.
- `continue_saved.jl`: resume a saved state with legacy or fine schedules.
- `measure_saved.jl`: CTMRG energy at a requested `chi`.
- `analyze_saved_series.jl`: per-stage energy and sector-resolved lambda report.
- `ctm_results.csv`: directly measured CTMRG results summarized above.

Large JLD2 states remain local under `results/` and are intentionally ignored
by git.  CSV, text reports, and configuration files are retained alongside
them.
