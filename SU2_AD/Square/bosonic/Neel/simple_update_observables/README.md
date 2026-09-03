# Square J1 SU(2) Simple-Update observables

Edit the configuration block in `Run_observables_J1_SU2_cell.jl`, then run:

```bash
julia Run_observables_J1_SU2_cell.jl
```

The input `T_set` is already the physical iPEPS.  `lambda_x` and `lambda_y`
are not absorbed again.  A single CTMRG environment is reused for:

- one-site density matrices and `<Sx>, <Sy>, <Sz>`;
- every nearest-neighbour two-site density matrix and `<S.S>` on x/y bonds;
- SU(2)-resolved transfer-matrix spectra in x/y directions;
- spin-spin correlations in x/y directions;
- parallel x-bond/x-direction and y-bond/y-direction dimer correlations,
  including raw and connected values.

The JLD2 output preserves metadata and structured results.  The optional MAT
output contains flattened numerical arrays for MATLAB analysis.  Dimer
separations start at two lattice spacings so that the two bond operators do
not overlap.  The CTM environment itself is not saved, so this analysis does
not create another large checkpoint.

The CTMRG truncation uses `truncdim(...; multiplet_tol=...)`; run this with the
project's configured `niusen/TensorKit.jl` fork.
