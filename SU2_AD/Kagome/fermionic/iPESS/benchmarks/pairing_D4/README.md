# Kagome Z2 iPESS: explicit onsite pairing benchmark

Uniform spinful kagome hopping, `t_up=t_dn=1`, `U=0`, `mu=-1`, and
`pairing=Delta` in `Delta*c_up†*c_down† + conj(Delta)*c_down*c_up`.
This is a quadratic Hamiltonian with an externally imposed pairing source,
not a self-consistent attractive-Hubbard mean-field calculation.
Particle-number parity is conserved; particle number is not.

The measured anomalous expectation is `F=<c_down*c_up>`.
For the real positive source used here, the exact F is negative.
Energies are per physical site, including chemical potential and pairing.

```powershell
julia --startup-file=no Kagome/fermionic/iPESS/tests/test_pairing.jl
julia --startup-file=no Kagome/fermionic/iPESS/benchmarks/pairing_D4/run.jl
julia --startup-file=no Kagome/fermionic/iPESS/benchmarks/pairing_D4/summarize.jl
```

The default run order is Delta=2,1,0.5, beginning with the largest gap.
Command-line arguments can select points, e.g. `run.jl 1.0`.
Saved complete method checkpoints are reused after checking model parameters.
Outputs are in `../../results/pairing_D4/`, separate from earlier benchmarks.

For each point, SU starts with seed 888 and even/odd virtual dimensions 2+2.
Its initial schedule is `(tau,dt)=(8,.1),(4,.02),(3,.005),(1,.001)`;
up to three further `(1,.001)` stages check whether successive measured
energies differ by less than `2e-7`.
FU and AD independently start from this same SU checkpoint.
FU uses three stages `(0.5,.05),(.2,.02),(.05,.005)`, up to eight ALS sweeps,
chi=12, and projector cutoff 1e-8.
AD uses L-BFGS with up to 50 iterations and gradient tolerance 1e-7.
Its chi, projector cutoff, and fixed CTM sweep count are selected using
finite-difference and independent energy checks; all attempts are logged.
This is unrolled CTMRG AD, with the environment response included.
The initial gradient check requires relative error <1e-3 or absolute error
<1e-7. The objective-to-measurement energy tolerance is
`max(2e-5, 0.001*abs(E_SU-E_exact))` per site. This tolerance controls the
approximation used for optimization, not the accuracy of final energies.

At Delta=0.5, the original absolute energy bound 2e-5 rejected all tested
settings: cutoffs 1e-8/1e-7 had unreliable initial gradients even up to
chi=32, while cutoff 1e-6 passed the gradient check but shifted energy by
2.49e-5. Extra CTM sweeps and finite-difference step sizes were checked.
The scaled energy tolerance above was introduced explicitly for this case;
the bias is 0.055% of its SU energy error. The rejected original checks are
preserved in `ad_initial_checks_absolute_bound.toml`, with further probes
in `ad_extra_checks.toml`. Exploratory perturbed states were not used:
the production AD starts from the original SU checkpoint.

Final measurements compare chi=16,32 at cutoff 1e-8 and chi=32 at 1e-10.
Only converged measurements enter the summary. All attempted measurements,
including convergence failures, remain in the per-point TOML files.
The selected D=4 Z2 sector dimensions, optimization status, density, pair
amplitude, and site nonuniformity are saved alongside the energy.

The exact reference uses each normal band xi=epsilon-mu:
`E_ground=sum(xi-sqrt(xi^2+abs(Delta)^2))/(3*Nk)`.
The two spins are already included in this expression.
It is checked against a 6x6 Nambu spectrum, a separate occupation-bit
three-site exact diagonalization, density/pair derivatives, and 300/600
momentum meshes. Quasiparticle gaps are Delta for the chosen chemical potential.
No claim of a globally optimal D=4 state follows from these finite runs.

`summarize.jl` also records energy gains at matched measurement settings,
checks physical onsite density-matrix bounds, and verifies the decomposition
into up/down kinetic, chemical-potential, and pairing energies.
`validate.jl` reblocks the SU and AD states around the opposite triangles;
their full-network energies, densities, and pair amplitudes are compared.
Generate the comparison figure with `python benchmarks/pairing_D4/plot.py`
from the `Kagome/fermionic/iPESS` directory.

Validation: `tests/test_pairing.jl` passed 110 assertions (52 pairing/BdG/
paired-optimization checks and 58 existing Hubbard-core checks). The exact
paired-trimer checks cover both triangle orientations, complex sources,
energy, density, anomalous expectation, and double occupancy.

## Results (2026-11-20)

Final energies per site, measured at chi=32 and projector cutoff 1e-10:

| Delta = BdG gap | Exact | SU | FU | AD |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | -1.1282095447 | -1.0831837988 | -1.0832198875 | -1.0834217864 |
| 1.0 | -1.3274689058 | -1.3083436499 | -1.3083571003 | -1.3084006143 |
| 2.0 | -1.9405787980 | -1.9365590173 | -1.9365597619 | -1.9365627207 |

| Delta | SU error | FU error | AD error | SU-FU | SU-AD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 4.50257459e-2 | 4.49896572e-2 | 4.47877584e-2 | 3.60887205e-5 | 2.37987589e-4 |
| 1.0 | 1.91252559e-2 | 1.91118056e-2 | 1.90682916e-2 | 1.34503348e-5 | 5.69643449e-5 |
| 2.0 | 4.01978069e-3 | 4.01903607e-3 | 4.01607727e-3 | 7.44628135e-7 | 3.70342246e-6 |

All three SU runs finished at total tau=17, with last-stage energy changes
below 4e-8. FU used total tau=0.75 after SU (90 local updates). FU stage
energies were not monotonic; the table uses the final small-step state.
Every AD run reached the 50-iteration limit, not the gradient tolerance.

| Delta | AD projector cutoff | Final gradient norm | Last 10 iterations' energy drop | Objective minus measured energy |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 1e-6 | 2.2671e-3 | 2.0022e-6 | 2.2497e-5 |
| 1.0 | 1e-7 | 3.2233e-4 | 1.3243e-7 | 7.0303e-6 |
| 2.0 | 1e-8 | 1.2651e-4 | 1.3395e-9 | 1.0243e-6 |

All AD objectives used chi=12 and four fixed CTM sweeps. The final directional
gradient checks had relative errors below 3e-5. All 27 final CTM measurements
converged, and every virtual bond retained even/odd dimensions 2+2.
All 270 FU local environments converged; metric extraction errors were
below 4e-16. Opposite-triangle reblocking of SU/AD states changed energy by
at most 2e-10. The onsite positivity bounds and energy decompositions passed.

At Delta=2 the tested measurement cutoffs shift absolute energies by about
1e-6, while the matched-setting SU-FU and SU-AD gains vary by less than
1e-9 and 1.3e-8 respectively. At Delta=0.5 these gain variations are below
5.4e-8 and 4.6e-9; at Delta=1 they are below 2e-13. These are observed
setting dependences, not rigorous error bounds.

Within this initialization and finite budget, AD gives the lowest energy
at every point, but its gain over SU remains small relative to the remaining
error against BdG. These runs do not establish a D=4 expressivity limit or
a global variational optimum. Density and pair amplitude,
sector dimensions, and all convergence diagnostics are in
`../../results/pairing_D4/comparison.toml`; the figure is `comparison.png`
in the same result directory.
