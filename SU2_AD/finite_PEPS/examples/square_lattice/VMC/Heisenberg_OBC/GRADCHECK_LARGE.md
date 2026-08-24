# Large-system VMC gradient validation

`verify_vmc_gradient_large.jl` follows the same contraction, sampling,
`contract_disk_derivative`, and Heisenberg local-energy path as
`mc_fPEPS_grad.jl`. It is designed for systems where a full coordinate finite
difference is impossible.

It performs two checks:

1. A deterministic per-configuration check of
   `d(log psi)/dt = sum_i O_i v_i` against the central difference of the
   contracted amplitude. This directly tests the hole derivative and is the
   first check to run.
2. An optional VMC energy check. It compares
   `2*real(<conj(E_local) O_v> - conj(<E_local>)<O_v>)` with a central
   difference obtained by correlated sampling on exactly the same reference
   configurations. Contiguous-block error bars are reported because this
   comparison is statistical.

The PEPS is normalized at most once before it is frozen. No perturbed PEPS is
normalized.

## Recommended server runs

Start with the cheap deterministic check:

```bash
PEPS_LX=10 \
PEPS_LY=10 \
PEPS_D=3 \
PEPS_CHI=6 \
PEPS_STATE=Heisenberg_SU_10x10_D3 \
PEPS_SCALAR=Float64 \
GRADCHECK_SAMPLES=8 \
GRADCHECK_AMPLITUDE_SAMPLES=8 \
GRADCHECK_DIRECTIONS=2 \
GRADCHECK_STEPS=1e-3,3e-4,1e-4 \
GRADCHECK_ENERGY_FD=false \
julia --project=/path/to/SU2_AD verify_vmc_gradient_large.jl
```

After the amplitude test reaches its expected central-difference plateau, run
the correlated energy check:

```bash
PEPS_LX=10 \
PEPS_LY=10 \
PEPS_D=3 \
PEPS_CHI=6 \
PEPS_STATE=Heisenberg_SU_10x10_D3 \
PEPS_SCALAR=Float64 \
GRADCHECK_SAMPLES=32 \
GRADCHECK_BURNIN_SWEEPS=100 \
GRADCHECK_SPACING_SWEEPS=10 \
GRADCHECK_BLOCK_SIZE=4 \
GRADCHECK_DIRECTIONS=2 \
GRADCHECK_STEPS=1e-3,3e-4 \
GRADCHECK_ENERGY_FD=true \
GRADCHECK_OUTPUT=/scratch/large_gradcheck.jld2 \
julia --project=/path/to/SU2_AD verify_vmc_gradient_large.jl
```

For a complex state set `PEPS_SCALAR=ComplexF64`. The direction parameter
`t` is always real even when the random direction is complex, so the predicted
directional derivative is `2*real(C_v)`.

## Reading the output

- `amplitude_relative_errors`: deterministic hole-gradient errors for every
  direction and step. They should decrease as `h^2` before roundoff or
  contraction noise creates a plateau.
- `predicted_derivatives`: VMC covariance predictions.
- `finite_derivatives`: common-sample correlated finite differences.
- `difference_standard_errors` and `z_scores`: block uncertainty of
  `finite_derivative - predicted_derivative`. With adequate sampling, the
  difference should be statistically compatible with zero and stable over a
  range of `h`.

The energy finite difference is intentionally expensive: for every direction,
step, sign, and sampled configuration it recomputes all bond-flip amplitudes.
Increase `GRADCHECK_SAMPLES` only after the deterministic check succeeds.
