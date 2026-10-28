# Convert the polarized SU(2) PESS and independently measure it

Entry point: `verify_charged_dense.jl`. It loads a charged SU checkpoint,
selects an auxiliary Sz component, converts every LOCAL tensor to an ordinary
dense array, and independently computes magnetization and Heisenberg energy.
An optional helper invokes the existing iPEPS_Z2 factorized dense CTMRG.
All files are new; existing ansatz, CTMRG and energy kernels are unchanged.

## What is projected

Conversion and projection are separate operations. The conversion expands
the full SU(2) TensorMap with `convert(Array,...)`. A subsequent
`selectdim(core,4,component)` selects the auxiliary index of the charged Tu.
With `component=1`, this is the highest-weight auxiliary component m_q=j.
No physical spin is measured or fixed individually by this operation.

An unprojected network has open auxiliary indices and does not specify one
unique physical-spin pure state. Including both physical and auxiliary
indices gives an invariant joint state. Tracing the auxiliary indices gives
an unpolarized physical mixed state; choosing an auxiliary boundary ket gives
a particular physical pure state. For many cells V_j tensor-powered N has
multiple total-spin channels. Choosing all auxiliary spins in |j,j> fixes
M_q=Nj, selecting the maximal J=Nj channel; this is the additional boundary
condition that selects the physical magnetization density 2j/9.

## Export and independent finite check

```sh
julia --project=. verify_charged_dense.jl /path/to/final.jld2 results/dense_check
```

The default h is 0.38, J1=1 and the BLAS thread count is 10. Use the same
TensorKit environment that wrote the input checkpoint. The following files
are saved:

* `dense_state.jld2`: projected, padded local arrays and cell map.
* `dense_state.toml`: portable complex128 arrays, original shapes, auxiliary
  projection, source SHA256 and the exact existing kagome convention.
* `dense_finite_report.toml`: independently measured finite-torus observables
  and differences from the original SU(2)-network reference.

The local kernels can leave different actual virtual dimensions. To use the
existing uniform-D dense ansatz, each virtual leg is embedded into the first
d components of a common dimension D, with remaining entries exactly zero.
The same embedding is applied at both ends of every bond. The script
explicitly contracts before and after padding and checks that the normalized
physical wavefunction is unchanged. No singular values are discarded.

The finite check contracts only ordinary dense local tensors on the nine-spin
periodic torus. Magnetizations and exchange energy are evaluated by explicitly
applying Pauli matrices. This is independent of the reference measurement's
bit-basis operator implementation. The auxiliary leg has already been removed.
For a controlled small calculation, exact finite verification requires D<=6.

## Infinite-lattice CTMRG in the same invocation

```sh
julia --project=. verify_charged_dense.jl /path/to/final.jld2 results/dense_ctm \
  --ctm --ipeps-root /path/to/iPEPS_Z2 --python python \
  --device cuda:1 --chi 8,12
```

`IPEPS_Z2_ROOT` can supply the package directory instead of `--ipeps-root`.
The helper `measure_dense_charged.py` requires Python>=3.11 and the existing
iPEPS_Z2 Python dependencies. It converts the portable TOML to the package's
standard JSON format, loads it through the existing dense ansatz, and calls
its existing Q/R factorized double-layer CTMRG and observable routines.

Defaults are cuda:1, 10 CPU threads, tolerance 1e-6 and at most 100 CTMRG
iterations. There is no silent device fallback. For a CPU measurement use
`--device cpu`; select cuda:0 explicitly on a machine with only that GPU.
The magnetic-field convention is H=H_exchange+h*sum(Sz), with normalized
magnetization m=-2*mean(Sz). The onsite field is included exactly once.

For each chi, `ctm/chi_N.json` records:

* energy, exchange energy and field energy, all per physical spin;
* normalized magnetization and site-resolved Sz on the nine independent spins;
* CTMRG convergence, corner-spectrum error and actual retained dimensions;
* variation between equivalent positions of the 3x3 computational grid;
* the field-energy reconstruction error and measurement settings/provenance.

The existing-state helper can also be called separately, without loading
Julia again:

```sh
python -B measure_dense_charged.py --state results/dense_check/dense_state.toml \
  --out results/new_ctm_check --ipeps-root /path/to/iPEPS_Z2 \
  --device cuda:1 --chi 16 --h 0.38
```

Use a new output directory for every invocation. No SU evolution is run by
either measurement entry point.

## Interpretation

The finite dense check should reproduce the finite SU(2)-network values.
Infinite CTMRG energies need not equal the finite nine-site torus energies.
The CTMRG convergence flag concerns the environment only: the input examples
were evolved for just six SU steps and are not optimized ground states.
Finite-chi truncation can also give a small deviation from the exact charge
density. Perfect m=1/9 remains selected by the auxiliary boundary condition;
it does not establish a stable plateau. See `CHARGED_CELL.md` for the SU
environment approximation and `DENSE_VALIDATION.md` for measured results.
