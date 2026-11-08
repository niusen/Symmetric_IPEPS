# Saved-U1 energy variance server batch

Run `sh run_energy_variance_server.sh` beside Project.toml in the server's
`Kagome/simple_update/u1_multicell` directory. The local repository calls that
directory `multicell`; all paths are relative, so either name works.
The shell submits one background Julia process, prints its log and PID, and
returns immediately. No Slurm, Python, GPU, watchdog or automatic restart.

## Calculation settings

All options are in `run_energy_variance_server.jl`:

- D=4 through 15; READ saved U1 states, never evolve or overwrite them.
- chi=40,60,80; CPU and 10 threads; periodic norm CTMRG at most 50 steps,
  tolerance 1e-8; two center-outward mixed-environment sweeps.
- Report cumulative variance for L=3,6,9. There is NO independent R setting.
- For each (D,chi), do ONE large-region contraction and extract all requested
  windows from its individually measured connected bond correlations.
- The contraction grid is automatically max(L)+6=15 on each side. The padding
  accommodates 2x2 observable patches, down-triangle bonds and the three source
  representative positions without wrapping a measured patch across the edge.
  This padding is not an independently scanned physical cutoff.

L is a square of primitive-cell ORIGINS; all six Hamiltonian terms assigned to
each origin are included, with the original Julia kagome conventions. Each
primitive position represents three physical spins. The original infinite
sqrt3xsqrt3 state/unit cell is unchanged. Windows are centered at each source
origin: offsets `-div(L-1,2):div(L,2)` in both directions (also supports even L).
All 18 inequivalent central bonds are summed and divided by nine physical spins.

This follows the LC-CTMRG large-region correlation contraction and extraction
of smaller sums described in arXiv:2511.22669v2, II.3 and III.1. It uses the
existing TensorKit explicit-projector/SVD backend, not the paper's iterative
projector solver. The default L values are an INITIAL screen, not a claim of
range convergence. Check outer-window increments/absolute sums before energy
variance extrapolation. Increase `variance_Ls` and use a NEW output root if the
outer contribution has not decayed. Increasing the actual contraction region
independently would require another contraction; small-L environment values
cannot simply be pasted into a larger mixed environment.

## Input states

The entry maps D=4..10 to:
`results/u1_from_D4_D4_D10_h038_chi40/D*/final.jld2`

D=11..15 to:
`results/u1_from_D10_D11_D15_h038_chi40/D*/final.jld2`

Every requested checkpoint must exist ON THE SERVER. CSV tables alone cannot
replace states. The shell checks paths without loading TensorKit or starting
CTMRG. The worker verifies the actual maximum virtual dimension matches D.

## Outputs and timing

Default root: `results/variance_D4_D15_chi40_60_80_L3_6_9_Lmax15`.
There are 36 (D,chi) contractions and 108 reported (D,chi,L) estimates.

- `summary.csv` / `variance_by_L_all.csv`: all estimates, energies, window sizes,
  signed/absolute shell contributions, largest shell correlation, imaginary
  parts, norm CTMRG status and actual retained corner dimensions.
- `timing_by_anchor_all.csv`: 18 central-bond timings per contraction, 648 rows.
- `D*/chi*/variance_by_L.csv`: cumulative results for the selected windows.
- `D*/chi*/connected_terms.csv`: every measured pair, offsets and source/target
  labels. Its `radius` column is only a distance diagnostic, NOT a task option.
- `D*/chi*/variance.json` / `variance.toml`: full data and numerical settings.
- `D*/chi*/timing_by_anchor.csv` and `run_timing.toml`: phases and UTC timestamps.
- `D*/chi*/completed.toml`: written LAST, denotes a finished contraction.
- `runs/<run-id>/D*.toml`: each D's full wall time for this invocation, including
  CTMRG, measurement and I/O. Earlier invocation records are preserved.
- `batch_completed.toml`: total invocation time and module-loading time.

The per-L rows share ONE contraction time. Do not sum their shared-time columns
as if three contractions were performed. Load/factor/norm timings have `_shared`
labels; source states are loaded/built once per D, norm CTMRG once per (D,chi).
First calls may compile Julia kernels. Timings are wall time, not peak memory.
If `/usr/bin/time` is available, the sh additionally writes whole-job wall/CPU
time and maximum resident set size to the end of the log. This is a process-wide
peak, not a separate peak for each D. All UTC timestamps explicitly end in Z.

Exchange Hamiltonian variance is reported. In an exact fixed-Sz U1 sector this
equals the variance with the uniform Zeeman term. CTMRG convergence alone does
not establish convergence in chi, spatial range or number of sweeps.

## Restart and uploads

Rerun the same sh after an interruption. Finished (D,chi) cases are verified and
skipped; an unfinished one is recomputed. Changed state hashes, numerical
settings or loaded implementation hashes are rejected in existing case folders.
Use a new output root when deliberately changing them. An exception stops the
batch with a traceback in the log; completed data are retained.

Upload the two root entry files and ALL seven `.jl` files in
`src/bosonic/kagome/native_measurement/energy_variance/`, retaining the relative
layout and the existing Julia library/Project/Manifest. No zip is required.
