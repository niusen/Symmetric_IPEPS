# Kagome inherited Hofstadter flux: D=4 benchmark

This model retains the triangular Hofstadter hopping coefficients at M=2,
t1=t2=1, and disconnects sites whose triangular coordinates (m,n) are both
odd. The remaining connected network is kagome. The disconnected sites are
excluded from its Hilbert space and per-site normalization. If retained,
they add a separate zero-energy atomic band per spin.

The parent primitive vectors are e1=(1,0), e2=(1/2,sqrt(3)/2). The actual
forward Hamiltonian coefficients are

* h(r+e1,r) = -1;
* h(r+e2,r) = -(-1)^m;
* h(r+e2-e1,r) = -i*(-1)^m.

These match `src/fermionic/square_Hubbard_model_cell.jl`'s
`get_Hofstadter_coefficients`, with m=px+1 in its bond convention.
No surviving bond is changed or regauged. Flux here is the phase of the
CCW product of these actual Hamiltonian coefficients, matching that code:
every parent triangle has pi/2. The kagome hexagon encloses six triangles,
so its flux is 3*pi = pi modulo 2*pi. A kagome cell encloses eight parent
triangles, with total flux 4*pi; no additional magnetic enlargement is needed.

Use kagome Bravais vectors A1=2e1, A2=2e2 and basis a=-e1, b=0, c=-e2.
In the kagome code's `H=-sum(t_ij*c_i†*c_j+h.c.)` convention, both up and
down triangles have `(t_ab,t_bc,t_ca)=(1,1,-i)`.
With k1=k.A1 and k2=k.A2, the independent off-diagonal Bloch entries are

```
h_ab = -(1 + exp(-i*k1))
h_bc = -(1 + exp( i*k2))
h_ca =  i*(1 + exp(i*(k1-k2)))
```

Hermitian conjugates fill the other entries; the diagonal is zero.
The eigenvalues are exactly

```
E_0(k) = 0
E_±(k) = ±sqrt(6 + 2*cos(k1) + 2*cos(k2) + 2*cos(k1-k2))
```

The lowest and highest bands span [-2sqrt(3),-sqrt(3)] and
[sqrt(3),2sqrt(3)]. Each is separated from the flat band by sqrt(3).
The Chern numbers from low to high are (-1,0,+1); reversing flux reverses
all Chern numbers. `bands.py` uses the gauge-invariant discretized
[Fukui-Hatsugai-Suzuki algorithm](https://arxiv.org/abs/cond-mat/0503172)
at meshes 30,60,120,300,600 and also checks the conjugated-flux model.
It independently builds the deleted-parent real-space Bloch matrix and
checks it against the three-band expression at random momenta.

## Energy benchmark

Use the existing spinful Z2 iPESS, with identical flux for both spins,
U=0, no explicit pairing, and mu=-sqrt(3)/2 in the lower band gap.
There are three bands PER SPIN (six counting the exact spin degeneracy).
The exact density is n=2/3 and double occupancy is 1/9.
The kinetic energy per kagome site is -1.6013551258806318; including
`-mu*n` gives -1.024004856691006. All optimizer energies include this term;
both kinetic and grand-canonical energies should be inspected if approximate
states have different densities.

SU was screened at seeds 777, 888 and 999, with virtual parity sectors 2+2.
Seed 999 gave the lowest SU energy and is the primary comparison; the original
seed-888 run is retained in the top-level results directory. Its initial (tau,dt)
schedule is (8,.1),(4,.02),(3,.005),(1,.001), followed by up to ten
(1,.001) stages; stop after stage>=5 if the last energy change is below 2e-7.
FU and AD independently start from the saved SU state.

FU optimizes the simplex and the three B tensors by up to 12 ALS sweeps
in each CTM-weighted overlap fit. It uses chi=16, CTM tolerance 1e-8,
projector cutoff 1e-8, and at most 80 environment sweeps. Its schedule is
(.5,.05),(.4,.02),(.2,.005), followed by up to three (.1,.005) stages;
after stage>=4 it stops if the last stage changes energy by less than 2e-7.
The primary seed-999 run adds a final (tau=.05,dt=.001) pass to match the SU
time step. A change of time step alone does not demonstrate convergence.
All stage checkpoints and energies are retained, since FU energies need
not be monotonic.

AD allows 150 L-BFGS iterations with gradient tolerance 1e-7, also saving
iterations 50 and 100. Initial chi, projector cutoff and fixed CTM sweeps
are selected using directional finite differences, sweep convergence,
and an independent CTM energy. The gradient check requires relative error
<1e-3 or absolute error <1e-7; allowed initial objective bias is
max(2e-5,0.001*abs(E_SU-E_exact)). Final states are independently remeasured.
At seed 999, cutoffs 1e-6 through 1e-8 failed the initial gradient check for
chi=16,24,32 despite converged energies. Cutoff 1e-5 passes at chi=16 with
four sweeps: directional relative error 1.7e-9 and initial objective bias
8.94e-5 relative to the strict CTM measurement. These failed settings are
retained in `seed999/ad_failed_initial_checks.toml`. No gradient tolerance
was relaxed to accept a failed setting.
The iteration cap is not a convergence claim or a guarantee of the global
D=4 minimum. Energy alone does not establish the optimized state's Chern number.

Run from SU2_AD:

```powershell
python Kagome/fermionic/iPESS/benchmarks/flux_D4/bands.py
julia --startup-file=no Kagome/fermionic/iPESS/tests/test_flux.jl
julia --startup-file=no Kagome/fermionic/iPESS/benchmarks/flux_D4/run.jl
```

For the primary comparison, in a loaded Julia session:

```julia
include("Kagome/fermionic/iPESS/benchmarks/flux_D4/run.jl")
directory=joinpath(FLUX_OUTPUT,"seed999")
run_flux_su(;seed=999,directory)
run_flux_fu(;directory,refine_tau=.05)
run_flux_ad(;directory)
include("Kagome/fermionic/iPESS/benchmarks/flux_D4/validate.jl")
validate_flux(;directory)
```

Then run `summarize.jl seed999` and `plot.py seed999`. `seed_check.jl`
reproduces the shorter initialization screen; `run_flux_su` continues its
checkpoint until the stated SU convergence check or stage limit.

After the run, call `validate_flux()` from `validate.jl` in the same loaded
Julia session, then run `summarize.jl` in a fresh Julia process and `plot.py`
with Python. Validation compares opposite triangle blockings and applies
an onsite U(1) gauge transformation to both the state and its hoppings.
It also remeasures saved AD iterations 50 and 100 at the final CTM settings.

The 38 assertions in `tests/test_flux.jl` passed: legacy-gauge matching,
independent occupation-bit Hamiltonians, band energies, complex isolated
trimers in both orientations, full-network reblocking, and AD checks.
The separate Python checks cover loop fluxes, surviving vertex degree,
parent-to-kagome Bloch reduction, analytic dispersion, Chern convergence,
and reversal of the magnetic flux.

Results and checkpoint files are in `../../results/flux_D4/`. All code
additions for this model are in the kagome directories; legacy triangular
source files are only read for validation.

## Results, 2026-11-20

Primary comparison: seed 999, D=4, all six virtual bonds retain parity
dimensions (2,2). FU and AD independently start from the same saved SU state.
The table uses converged CTM measurements at chi=32, projector cutoff 1e-10,
and environment convergence tolerance 1e-8. Energies are per kagome site.
`E` includes `-mu*N`; kinetic energies at different measured densities should
not be interpreted as a canonical comparison at fixed filling.

| State | E | E-E_exact | Kinetic energy | Density |
|---|---:|---:|---:|---:|
| Exact | -1.024004856691 | 0 | -1.601355125881 | 0.666666667 |
| SU | -0.816304888983 | 0.207699967708 | -1.390074682566 | 0.662532290 |
| FU final | -0.816140938790 | 0.207863917901 | -1.394347684628 | 0.667655641 |
| AD, 150 iterations | -0.817130682679 | 0.206874174012 | -1.394066154396 | 0.666187700 |

The best saved FU stage is stage 1 (tau=.5, dt=.05), independently remeasured
as E=-0.816429637681. It improves SU by 1.24749e-4; later FU stages increase
the energy. The final FU value is worse than SU by 1.63950e-4. AD improves
SU by 8.25794e-4. None of these results demonstrates a high-accuracy D=4
ground state: errors relative to the exact grand energy are about 20.2%.
They do not establish the globally optimal D=4 energy or a ranking of
fully converged optimization methods.

Convergence and numerical qualifications:

* SU reaches total tau=22; its last tau=1 stage changes the measured energy
  by -9.82e-8. This only tests stability within that SU basin.
* FU performs 180 second-order time steps, 540 local triangle updates,
  total additional tau=1.45. It does not pass the stage energy convergence
  check. All 540 CTM environments converge; maximum metric extraction error
  is 6.79e-16, Hermiticity error 4.11e-14, and discarded negative metric
  weight is zero.
* On both the initial SU state and the final FU state, a separate dt=.005
  local fit stops after 3 ALS sweeps. Caps of 12 and 40 give the same loss
  and fidelity. The normalized full metric differs from the product of
  its one-leg marginals by about 1.9% in relative Frobenius norm. These
  checks confirm a nonproduct CTM metric and iterative overlap fitting;
  they do not imply global energy convergence.
* AD ends with `status=maxiter`, gradient norm 0.00415, above 1e-7. Its
  final optimization objective is -0.817172944668, 4.23e-5 below the strict
  energy measurement. The final directional check has relative error
  2.14e-6. All 12 checks over two directions, two difference steps, and
  4/8/12 CTM sweeps pass. The energy spread over sweeps is 4.36e-8;
  the relative gradient change from 8 to 12 sweeps is 5.61e-7.
* The strict measured AD energies at iterations 50, 100, 150 are
  -0.817128016054, -0.817128194807, -0.817130682679. Thus much of the late
  objective decrease reflects changing projector truncation bias. Attempts
  to tighten the final-state AD cutoff to 1e-6, 1e-8, 1e-10 at chi=16,
  eight sweeps still FAIL directional checks (relative errors .122, .405,
  .256). Changing chi to 8,10,12 at cutoff 1e-8 also fails. These settings
  were not used for further optimization. Stable differentiation of the
  more accurate contraction remains unresolved; the valid 1e-5 objective
  gradient must not be confused with a verified gradient of the strict
  contraction.

All final measurement settings converge. Their energy spreads are at most
3.10e-7. Opposite triangle blockings differ by less than 4.69e-9 for all
three states; the AD onsite gauge-covariance energy difference is 8.89e-15.
The states have nonzero anomalous pairing expectation even though the
Hamiltonian conserves particle number and the exact ground state has zero
pairing expectation. This is a limitation of the approximate Z2 states,
not evidence for superconductivity in this free model.

Initialization matters substantially. The strict SU energies for seeds
777, 888, 999 are -0.744436911287, -0.765873467378, -0.816304888983.
The seed-777 screening run had not converged. The original seed-888 AD run
stopped at iteration 48 with `line_search_failed` and FAILED its final
gradient check (relative error .312). Its top-level result files are
retained as diagnostics, not presented as converged AD accuracy.

Primary machine-readable results: `../../results/flux_D4/seed999/comparison.toml`.
Figures: `../../results/flux_D4/bands.png` and
`../../results/flux_D4/seed999/comparison.png`.
To reproduce the supplemental checks, include `ad_audit.jl` and `fu_audit.jl`,
then call `audit_flux_ad`, `probe_flux_ad_truncation`, and `audit_flux_fu`
(both `method="su"` and `method="fu"`) with the same results directory.
