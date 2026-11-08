# Kagome native CTMRG memory fix

The failing D12 job finished all simple-update stages. Its failure was in
`factor_corner(:RU,3)`, allocating a rank-5 environment/Q/R contraction
temporary. The original contraction order first assembled the environment,
then attached one rank-3 local factor. Its dense shape scales as chi² D⁶.

`src/bosonic/kagome/native_measurement/memory_bounded_ctm.jl` adds specialized
TensorKit methods with exact reordered contractions for all corner rotations,
neutral/charged insertions, and edge updates. Original generic methods remain
available for reference. For uniform fused legs delta=D², the largest new
corner/edge intermediate has at most chi² delta² elements (a 1D charged
spectator does not increase this). This is a shape analysis, not a promise of
total process RSS; U1 blocks, temporary permutations, initial boundary sizes,
and the CTMRG projector/SVD also contribute.

As in the triangle implementation, `CTM.get_AA_direction` can build one site's
rank-4 double-layer AA on demand. Production absorption retains Q/R factors
and closes them with the boundary in the smaller order instead of forming AA
or a rank-5 temporary. No full-cell AA cache is introduced.

The same file also forms upper/lower CTMRG half-planes separately to shorten
enlarged-corner lifetimes, and streams observable insertions rather than
caching all Sz/S+/S- enlarged corners together. It changes neither state,
truncation settings, initial environment, physical convention, nor bond
counting. It adds no garbage-collection call to every contraction.

## Server update and resume

Update these two files in the existing repository, preserving their paths:

- `src/bosonic/kagome/native_measurement/memory_bounded_ctm.jl` (new)
- `src/bosonic/kagome/native_measurement/KagomeNativeMeasurement.jl` (loads it)

Restart the same D10-to-D15 launcher with the same settings/output directory.
The existing continuation workflow skips completed D11 measurements and uses
the already completed D12 `final.jld2` when its normal completion checkpoints
are present. It then retries D12 CTMRG and observables, not the 9000 SU steps.
Do not remove completed states or locks, and do not run two writers at once.
An already running Julia process does not acquire the new code automatically.

The new result has `contraction_backend = "memory_bounded_QR_v1"`.
The measurement settings/cache key remains compatible: old completed results
are not invalidated, since the new contraction is algebraically identical.
Original D10 and all other state files are read only during these tests.

## Small-D verification

`Kagome/simple_update/multicell/tests/test_memory_bounded_ctm.jl` loads the
original method definitions first (even with the patch installed). It checks:

- The same complex random dense D2 state and saved optimized U1 D4 state.
- Old/new complete CTMRG, chi=4, max 12 iterations, tolerance 1e-6, h=0.38.
- Also the old/new observables in exactly the same old CTM environment.
- Total/exchange/field energies, all site Sz, every up/down bond, and both
  orientations of SzSzSz; geometry labels, bond sums and field normalization.
- All 16 corner orientations, neutral/charged insertions, and all four edge
  directions against both old expressions and an explicit local-AA reference.
- Warmed allocations of the failing corner topology at small D4, chi=4.

Results are written to `memory_bounded_ctm_checks.toml`. No SU evolution or
large-D calculation is required. The fixture path and report folder can be
overridden with KAGOME_TEST_ROOT and KAGOME_MEMORY_TEST_OUT for a staged test.

## Recorded checks (2026-11-02)

TensorKit 0.14.11 / Julia 1.11.6: all 503 assertions passed. Both old and new
methods converged after 7 CTMRG iterations at chi=4, tolerance=1e-6.

| Same fixed state | Total energy difference | Max site Sz difference | Max bond difference | Max triangle SzSzSz difference |
|---|---:|---:|---:|---:|
| Complex random dense D2 | 6.59e-16 | 3.71e-14 | 1.71e-14 | 2.20e-15 |
| Saved optimized U1 D4 | 0 | 7.63e-16 | 5.55e-16 | 1.04e-16 |

The warmed failing-corner topology at D4, chi4 allocated about 2.37 MB with
the original order and 0.35 MB with the new order, about 85% less. These are
total per-call allocation counts, not peak resident memory. Symbolic tests
also check all 36 neutral/charged corner and edge expressions at D12 chi40,
initial boundary dimension144, and D15 chi40 without allocating those tensors.
This establishes small-D numerical equivalence, not a large-D RSS measurement.

TensorKit 0.17.0 / Julia 1.11.6: the same 503 assertions also passed, with both
states again converged after 7 iterations. Maximum old/new observable
differences were 2.89e-14 (dense D2) and 1.20e-15 (U1 D4). Warm per-call
allocations were 2,405,153 bytes old vs 380,104 bytes new. No package lockfile
or installed global environment was changed for the compatibility checks.

Full logs and TOML values are retained in
`Kagome/simple_update/multicell/results/validation/ctm_memory_20261102/`.
