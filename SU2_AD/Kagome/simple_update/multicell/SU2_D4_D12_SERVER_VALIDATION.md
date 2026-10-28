# Pure Julia/TensorKit validation, 2026-10-28

No D4..12 production scan was executed locally. All numerical checks below
use Julia1.11.6 and CPU with two BLAS threads. No Python numerical process
is called by the implementation or its Julia tests.

## Conversion and physical observables

`test_native_measurement.jl` passes **92/92** assertions with TensorKit
**0.14.11**, and **92/92** with TensorKit **0.17.0**. Checks include every
local tensor/lambda coefficient in the U1-to-dense conversion, compatible
truncated SVD/inverse projectors, both native CTMRG representations,
energy/field/magnetization, nine site values,18 bonds and six SzSzSz triples.
The D2 state is initialized with the existing SU2 integer-singlet parity
initializer and evolved for only two original Trotter steps.

At h=0.38 and chi4, both U1 and dense give:

* E/spin=-0.35444444444444445 (rounding difference below2e-16).
* m=-2*mean(Sz)=0.1111111111111111.
* Field contribution=-0.38/18; exchange energy=-1/3.
* CTMRG converges after two iterations. The18 bond values reconstruct the
  exchange energy, and all six raw triples are within the spin1/2 bound1/8.

## Nontrivial D4 reference-state measurement

The bundled previously evolved SU2 D4 state was measured with **native U1
CTMRG, chi8, cap50 and tolerance1e-6**, without further SU evolution.
CTMRG converged in **10 iterations**, final error **4.851327445454095e-7**.

* Native Julia E/spin: **-0.4336936696485817**.
* Earlier Python measurement of this SAME state at chi8:
  **-0.4336936696485819**.
* Difference: **2.220446049250313e-16/spin**.
* Native normalized magnetization: **0.11110976813810049** at this small chi.

This checks the change of measurement backend. It does not assert equality
between independently optimized SU2 and unrestricted dense ground states.
The same-state check also exercises nonzero transverse exchange, the
nonuniform virtual spaces, auxiliary projection and factorized environment.

## Server I/O and recovery

`test_native_server_workflow.jl` passes **13/13** checks with TensorKit0.14.11.
A D2 seed is allowed one SU step with capD4, followed by only two chi2 CTMRG
iterations. It checks saved JLD2/TOML/JSON observables, aggregate CSV files,
resume/skip with unchanged report hash, and rejection of changed settings.
This deliberately short measurement is marked unconverged and is not a
production D4 optimization result. JSON/TOML values and CSV row counts
were independently checked. The submission sh passes POSIX syntax checking.
The installed default plan has D4..12, exactly9000 steps/D, chi40, cuda:1,
CTMRG cap50 and tolerance1e-6. No production evolution was launched.

## Reused code, Python audit and GPU scope

The original SU2/U1 kernels are unchanged. Native functions reuse the
existing CTM structures/settings, coordinate maps, projector construction,
strip update and CUDA storage helpers. Explicit contraction order keeps
Q/R double layers separate, including initialization and operator insertion.
The rank4 onsite D^8 double layer is never formed.

The Python numerical iPEPS_Z2 package had no tracked modifications from
this SU2 work and was not rewritten. The SU2-only bridges
measure_dense_charged.py, u1_ctm_backend.py, measure_su2_server.py and
test_su2_server_measurements.py were removed from the Julia package.
The previously requested Matplotlib visualization helper is retained;
it is not part of the numerical workflow.

GPU device/storage code reuses the repository's CUDA/cuTENSOR/Adapt helpers.
Actual cuda:1 execution was not tested locally; GPU startup checks the
selected device and never silently falls back to CPU. CPU and GPU projects
are selected by the pure Julia bootstrap from the user's settings.
