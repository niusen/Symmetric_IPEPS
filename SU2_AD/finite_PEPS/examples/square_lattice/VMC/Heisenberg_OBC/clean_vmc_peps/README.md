# Clean VMC PEPS Reference

This folder contains a small dense PEPS + VMC reference implementation for
checking gradient conventions.

It deliberately avoids `TensorKit`, symmetry sectors, MPS truncation, and Markov
chain state recycling.  The target case is a small square-lattice OBC PEPS, by
default `4x4, D=2`.

## Run

```powershell
& "C:\Users\senni\.julia\juliaup\julia-1.11.6+0.x64.w64.mingw32\bin\julia.exe" `
  "C:\Users\senni\Documents\Codex\2026-05-14\d-my-documents-code-julia-codes\clean_vmc_peps\clean_vmc_peps.jl"
```

## What It Checks

1. Exact PEPS amplitude contraction by row transfer matrices.
2. Hand-written amplitude derivative for one fixed spin configuration.
3. Exact VMC energy and gradient by enumerating all `Sz=0` configurations.
4. Finite-difference checks of randomly selected PEPS tensor entries.
5. Direct-sampling MC convergence of the covariance gradient estimator.
6. Metropolis Markov-chain sampling convergence using nearest-neighbour
   spin-exchange moves.

## Gradient Formula

For real PEPS parameters and sampling probability `p(s) = psi(s)^2 / Z`,

```julia
Grad = 2 * (mean(E_loc(s) * O(s)) - mean(E_loc(s)) * mean(O(s)))
O(s) = d psi(s) / psi(s)
```

The factor `2` is required because the sampling weight is `psi^2`.

## Current Verification Results

For the default `4x4, D=2` random PEPS:

```text
Worst exact-gradient FD relative error = 7.63e-6

Direct independent sampling:
MC nsamples=  200: grad overlap 0.723128, relnorm 8.432e-01
MC nsamples= 1000: grad overlap 0.907980, relnorm 4.638e-01
MC nsamples=10000: grad overlap 0.970311, relnorm 2.430e-01

Metropolis Markov chain:
Markov nsamples=  200: grad overlap 0.477720, relnorm 1.579e+00
Markov nsamples= 1000: grad overlap 0.849875, relnorm 5.975e-01
Markov nsamples=10000: grad overlap 0.953867, relnorm 3.007e-01
Markov nsamples=50000: grad overlap 0.989854, relnorm 1.478e-01
```

Run the longer Markov-chain check with:

```powershell
& "C:\Users\senni\.julia\juliaup\julia-1.11.6+0.x64.w64.mingw32\bin\julia.exe" `
  "C:\Users\senni\Documents\Codex\2026-05-14\d-my-documents-code-julia-codes\clean_vmc_peps\run_markov_long.jl"
```
