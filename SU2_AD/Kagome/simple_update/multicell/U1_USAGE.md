# U(1) conversion and Simple Update

`verify_charged_dense.jl` now exports **U(1) by default**. Its historical
filename is retained so existing commands continue to work. Dense export is
available with `--representation dense`. The SU runner keeps its SU(2)
default; choose `--symmetry u1` to evolve genuine U(1) TensorMaps.

## Convert an existing polarized SU(2) checkpoint

Run in this directory with the TensorKit environment that wrote the input:

```sh
julia --project=. verify_charged_dense.jl /path/to/final.jld2 results/u1_check
```

Outputs are `u1_state.jld2`, `u1_state.toml`, and `u1_finite_report.toml`.
The JLD2 file contains TensorKit U(1) block tensors and all SU lambda tensors.
The portable TOML records coefficients, charge sectors, arrows, and cell map
for the Python block U(1) CTMRG reader. It is a transport format, not the
in-memory tensor representation. An already projected U(1) checkpoint can
be exported through the same command.

The independent nine-site finite verification in this entry point is limited
to a single charged sqrt3 cell with actual D<=6. The `to_u1` library function
and SU runners have no such finite-torus limit; runners save portable output
directly, which can be passed to the Python measurement helper.

To request the previous dense format explicitly:

```sh
julia --project=. verify_charged_dense.jl /path/to/final.jld2 results/dense_check --representation dense
```

Charges use **q=2Sz**, so a physical spin has labels +1,-1. Restriction
expands each SU(2) multiplet into its U(1) components, preserving degeneracies
and dual arrows and reordering to TensorKit's actual U(1) basis. For a charged
SU(2) checkpoint, `--component 1` selects the auxiliary highest weight.
The auxiliary leg becomes a one-dimensional charged leg, retained in Julia
to make each tensor an invariant intertwiner. This is the same physical
projection as the previous dense export; no singular values are discarded.

## U(1) Simple Update

A small polarized trial, initialized by SU(2) projection, takes six full
Trotter steps by default:

```sh
julia --project=. run_charged_trial.jl results/u1_trial --symmetry u1 --Dmax 3
```

To start from your existing SU(2) polarized checkpoint instead:

```sh
julia --project=. run_charged_trial.jl results/u1_from_su2 --symmetry u1 --init /path/to/su2_final.jld2 --Dmax 3
```

`--schedule '0.01:0.02,0.003:0.006,0.001:0.002'` optionally specifies
dt:tau pairs. This trial measures an exact finite nine-site torus after
evolution; use the general runner below for larger D or independent 3x3 cells.

The general runner supports both cell choices, the requested 9000-step
production schedule, and stage checkpoints:

```sh
# Small workflow check only:
julia --project=. run.jl --symmetry u1 --cell sqrt3xsqrt3 --Dmax 3 --init /path/to/su2_final.jld2 --smoke --out results/u1_smoke

# Production commands, not run in local validation:
julia --project=. run.jl --symmetry u1 --cell sqrt3xsqrt3 --Dmax 6 --init /path/to/su2_final.jld2 --out results/u1_sqrt3_D6
julia --project=. run.jl --symmetry u1 --cell 3x3 --Dmax 6 --init /path/to/su2_final.jld2 --out results/u1_3x3_D6
```

For `3x3`, a sqrt3 input is copied onto nine independent representatives.
The initial periodic state is unchanged, with three charged auxiliary legs
in the 27-spin cell. Subsequent updates do not tie these representatives.
`--init` accepts charged SU(2), ordinary SU(2), or U(1) checkpoints.
Without `--init`, the general runner restricts an ordinary SU(2) singlet
initialization: that is a zero-magnetization sector, **not** the 1/9 state.
`--component N` chooses the projected component of a charged SU(2) input.

The initial SU(2) space defaults to 0 plus 1/2 (D=3). U(1) SU truncates
individual charge states and can therefore use Dmax=2 as well. Dmax counts
states, not sectors; the initial checkpoint may be larger until the first
complete update. U(1) truncation has no SU(2) multiplet-boundary removal.
The original triangle HOSVD and down-half/(up,down)/up-half sequence are
retained. All lambda weights remain checkpointed. `--symmetry su2` retains
the previous SU(2) behavior.

Resume from a completed U(1) stage into a new output directory using
`--resume stage_02.jld2` instead of `--init`. Keep the same cell, Dmax, J1
and schedule (`--smoke` too, if applicable).

## Block U(1) CTMRG and observables on the GPU

```sh
python -B measure_dense_charged.py --state results/u1_sqrt3_D6/u1_state.toml --out results/u1_ctm --ipeps-root /path/to/iPEPS_Z2 --device cuda:1 --chi 40 --h 0.38
```

This automatically selects block U(1) for U(1) input, or the original dense
backend for explicit dense input. It retains the Q/R iPESS factorization;
the double layers, moves, and observables are U(1) block tensors. Exchange
uses SzSz+(S+S-+S-S+)/2 so charged transverse operators are handled correctly.
Only small corner spectra are expanded for the convergence diagnostic.
Defaults remain 10 threads, tolerance 1e-6, at most 100 CTMRG iterations,
and cuda:1. Select `--device cpu` or another GPU explicitly when needed.
Each chi saves JSON with energy, site Sz, magnetization, convergence and
actual corner dimensions. No silent CPU or dense fallback is used.

H=H_exchange+h*sum(Sz), with normalized magnetization m=-2*mean(Sz).
The field is included once in measurement. In a fixed-charge sector it is
a constant, so SU evolves the exchange gates. Fixed m=1/9 is a sector
constraint, not evidence of a finite plateau width; sector-energy
comparisons are still required. U(1) evolution preserves Sz but does not
enforce SU(2) multiplets or fixed total S^2.

## Tests

```sh
julia --project=. test_u1_cells.jl
julia --project=. test_u1_initialization.jl
```

See `U1_VALIDATION.md` for the tested versions and numerical comparisons.
