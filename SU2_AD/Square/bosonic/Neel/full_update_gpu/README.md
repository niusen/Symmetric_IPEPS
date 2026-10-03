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
  the compute device. Each rotated tensor is built on that device individually.
- `ctm_offload_intermediates`: park completed `MMup`, `MMlow`, reflected halves,
  and `RMup`/`RMlow` on CPU while they wait for their next use. Restore only the
  factors needed for the next product or projector construction.

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
GPU pool usage is reported separately. Diagnostic output is off by default.

The GPU folder also contains CuArray-specialized versions of the existing
CTMRG initialization, closed double-layer, and open double-layer builders.
Their contraction order is copied from the CPU routines; only TensorKit's
temporary `unitary` maps are moved to storage matching the input tensor, as in
the triangular iPESS GPU code.  Their scalar type is matched as well, because
cuTENSOR does not perform the CPU code's implicit `Float64`/`ComplexF64`
promotion.  The Heisenberg gate and energy operator receive the same treatment.

The server environment needs `CUDA.jl`, `cuTENSOR.jl`, and `Adapt.jl`, plus a
TensorKit version with CUDA/cuTENSOR extensions. Device strings are `"cpu"`,
`"cuda"`, `"cuda:0"`, `"cuda:1"`, and so on.

Saved files always contain CPU-backed TensorMaps and can therefore be loaded
by either the existing CPU runner or this GPU runner.
