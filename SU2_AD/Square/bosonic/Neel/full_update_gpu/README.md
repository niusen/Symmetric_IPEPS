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
