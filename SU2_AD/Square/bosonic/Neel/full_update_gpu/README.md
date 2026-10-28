# Square-J1 GPU Full Update

Run `Run_full_update_J1_SU2_cell_GPU.jl` after editing its configuration block.
No command-line parameters are required.

The implementation follows the triangular Hofstadter-Hubbard GPU Full Update:

- the persistent iPEPS and JLD2 checkpoints remain on CPU;
- CTMRG, local Full Update, and energy measurement can use separate devices;
- tensors are moved to the selected device only for that stage and are moved
  back to CPU before the CUDA memory pool is reclaimed;
- the local Full Update precontracts the CTM and fixed rank-4 residual tensors
  into a positive metric, then updates the two rank-3 bond tensors with ALS
  linear solves; it does not differentiate through the tensor network;
- CTMRG is reconstructed from scratch after every updated bond, exactly as in
  the CPU square-J1 Full Update.

All three stages use the selected GPU by default.  To reduce GPU-memory
pressure, set any of `ctm_device`, `full_update_device`, or
`observable_device` to `"cpu"` independently in the configuration block.

CTMRG also has two independent CPU-parking switches (both enabled in the runner):

- `ctm_offload_double_layer`: retain closed AA tensors, fusion maps, and the four
  rotated AA caches in CPU RAM; load only the current direction's AA cell onto
  the compute device. Upload each site's source AA once to build its four
  rotations, downloading each result before building the next one.
- `ctm_offload_intermediates`: park completed `MMup`, `MMlow`, reflected halves,
  and `RMup`/`RMlow` on CPU only when their symmetry-block data size is at least
  `ctm_intermediate_min_mib` (default 256 MiB). Smaller tensors stay on GPU.
  A value of 0 parks all sizes; it does not restore the previous excessive
  copying or garbage-collection schedule. This is a size policy, not an
  automatic free-memory test; lower the threshold if memory is tight.

`RMup` is normalized before its single CPU download and that host copy is reused
after forming M and after SVD. `RMlow` stays on GPU until M has been formed, then
is downloaded at most once, for the SVD interval. Small GPU-resident factors
are reused directly, without an Adapt round trip. All normalization, products,
permutations and SVD remain on the compute device.

Full GC and CUDA pool reclamation happen at direction/stage boundaries (also
after building each site's original AA and after preparing the rotation cache),
not inside the projector loop. Releasing large intermediates uses a synchronized
incremental GC without clearing the CUDA pool. Small intermediates do not
trigger explicit GC. Pool-reserved memory can therefore stay above live tensor
memory inside a direction; that alone does not imply a leak.

These switches use the separate `src/bosonic/square/gpu/CTMRG_unitcell_offload.jl`,
copied from the original CTMRG file with renamed entry points. Contraction
indices/order, projector truncation, PBC initialization, direction order, and
convergence/plateau criteria are preserved. The original CTMRG file is unchanged.
Contractions and SVD still run on `ctm_device`; no automatic differentiation is
used. Disable both switches to use the copy's original all-device storage path.
When `ctm_device="cpu"`, parking is automatically inactive.

Parking trades host RAM and CPU/GPU transfers (plus explicit garbage collection)
for lower simultaneous GPU residency. It does not bound a single contraction's
workspace, so it cannot guarantee that every D/chi combination will fit.
Set `ctm_print_memory=true` for diagnostic runs: it reports closed-AA data size,
initial C/T data size and edge boundary dimension, direction memory snapshots,
and projector-intermediate data sizes. PBC initialization is unchanged and its
initial boundary dimension can exceed the requested chi before truncation.
Tensor data sizes exclude contraction/SVD workspaces and allocator bookkeeping;
GPU pool usage is reported separately. The runner currently enables diagnostic
output (`ctm_print_memory=true`); set it to false for quiet production runs.

The GPU folder also contains CuArray-specialized versions of the existing
CTMRG initialization, closed double-layer, and open double-layer builders.
Their contraction order is copied from the CPU routines; only TensorKit's
temporary `unitary` maps are moved to storage matching the input tensor, as in
the triangular iPESS GPU code.  Their scalar type is matched as well, because
cuTENSOR does not perform the CPU code's implicit `Float64`/`ComplexF64`
promotion.  The Heisenberg gate and energy operator receive the same treatment.

Energy measurements select their contraction strategy in the runner:

- `energy_method = :reduced` (runner default): the separate
  `src/bosonic/square/gpu/square_J1_bond_energy_reduced.jl` reuses the exact
  FU site split and the residual-to-CTM half-contraction order. It does NOT
  run an FU step: no evolution gate, Dmax truncation, ALS solve, Hermitian
  symmetrization, eigenvalue clipping, or positive-metric factorization.
  The original raw halves are joined to form E, and the small two-site
  tensor B gives the physical density `rho = B' * E * B`.
- `energy_method = :legacy`: original AA_open measurement. The runner uses
  its streaming 2x1/1x2 implementation as a small-D reference.

The reduced path never constructs AA_open. It transfers CTM and the existing
U_L/U_D/U_R/U_U fusion maps to the observable device, not the closed AA cell.
It supports `observable_device="cpu"` as well. Non-inlined stages hold the
current tensors in a per-bond workspace; consumed slots are cleared and GC is
requested after each stage returns. Only the scalar energy escapes a bond
measurement. With `print_gpu_memory=true`, diagnostics show explicit workspace
data and device memory after reclamation (not the peak inside a contraction).
There is no manual `unsafe_free!` or new CPU offloading of intermediates.

The mode is printed and saved in checkpoints. Low-level energy APIs still
default to `:legacy`; select `energy_method=:reduced` explicitly outside the
runner. `low_memory` selects streaming for legacy only; reduced is always staged.
The experimental symmetric-AA_open builder was removed along with its include
and `aa_open_method` option. Update the runner and source files together.
Original CPU/GPU AA_open builders, CTMRG, and FU optimization remain unchanged
by this measurement implementation.

Run `Run_test_energy_methods_J1_SU2_cell.jl` independently for comparison; it
does not run any optimization. Settings are at the top of that file:

- default 2x2 cases: mixed D=3 and minimal x/y-staggered parity patterns with
  maximum D=4, using independent real/imaginary parts;
- `run_device="cpu"` by default, configurable to a CUDA device;
- `initial_state_file` optionally loads a checkpoint next to the test script
  (or an absolute path); `maximum_test_D=4` guards against a large legacy OOM;
- exactly one CTMRG per case, shared by the two energy methods;
- comparison of every x/y bond and total energy, with printed differences,
  tolerance checks and a bitwise-equality indicator;
- a timestamped JLD2 comparison report (no input state is overwritten), and
  an error exit if any bond or total energy fails the tolerance check.

Finite-chi CTM convergence is reported separately: both methods must agree on
the same network even when that CTM has not converged. Numerical equality is
tested with tolerances because floating-point contraction orders differ.
No local Julia/GPU execution was performed. The comparison and large-D memory
behavior still need verification on the configured server.

The server environment needs `CUDA.jl`, `cuTENSOR.jl`, and `Adapt.jl`, plus a
TensorKit version with CUDA/cuTENSOR extensions. Device strings are `"cpu"`,
`"cuda"`, `"cuda:0"`, `"cuda:1"`, and so on.

Saved files always contain CPU-backed TensorMaps and can therefore be loaded
by either the existing CPU runner or this GPU runner.
