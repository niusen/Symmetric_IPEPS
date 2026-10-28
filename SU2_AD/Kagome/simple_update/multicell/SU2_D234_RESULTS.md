# Charged SU(2) D=2,3,4 convergence checks, 2026-10-28

All results use sqrt3xsqrt3 (nine physical spins), J1=1, h=0.38,
H=H_exchange+h*sum(Sz), and normalized magnetization m=-2*mean(Sz).
An auxiliary spin 1/2 on Tu(1) is projected to its highest weight.
The evolution is **SU(2) Simple Update**, not gradient variational
optimization of the infinite projected-state energy. Original local
triangle kernels, multiplet truncation and Trotter order are unchanged.

## Selected stable candidates

These are the lowest-energy stable SU outputs among the starts tested here,
not established global minima of the corresponding SU(2) tensor families.
The energies below are infinite-system U(1) CTMRG energies per physical spin,
not nine-spin finite-torus energies.

| Dmax | Selected start | chi | Energy/spin | m | CTM error |
|---:|---|---:|---:|---:|---:|
| 2 | half-integer legs plus two singlet copies on integer legs | 24 | -0.3544444444444445 | 0.1111111111111111 | 0 |
| 3 | uniform real reduced coefficients, 0 plus 1/2 initial space | 33 | -0.3642942148193873 | 0.1111111145845544 | 4.30e-7 |
| 4 | explicit integer/half-integer pattern, initially 0 plus 1 versus 1/2 | 33 | -0.4336944413459763 | 0.1111111117269896 | 3.63e-7 |

CTMRG tolerance is 1e-6, capped at 100 iterations; selected results converged
in 2, 8, and 10 iterations respectively. Ten CPU threads were used. Both
the existing GPU and CPU measurement backends were used locally; no Python
source was modified for this calculation. U(1) input uses genuine block
tensors. Dense calculations were explicitly selected as small-D diagnostics.

### D=2: 1/2 versus 0 plus 0 on different bonds

This is a valid SU(2) assignment. It is different from putting spin 1/2 on
every leg: two virtual halves at B cannot couple to its physical half,
and an ordinary simplex with three half-integer legs cannot be a singlet.

The explicit pattern uses one half-integer and one integer leg at every B.
With integer space 0 plus 0, every virtual leg initially has dimension two.
Nevertheless the B matrix unfolding has rank **one** on each singlet-copy
leg: there is only one fusion channel to its single half-spin partner.
The orthogonal singlet-copy combination is physically unused.

Numerical checks show that all nine integer legs compress from dimension
two to one after the first complete update. The normalized nine-spin
wavefunctions with 0 and with 0 plus 0 agree to 3e-12 before and after that
update. Both constructions reach E=-0.3544444444444445, m=1/9. The selected
state consists of nearest-neighbor singlets and a polarized spin; its
CTM corners have dimension one. Adding a second singlet copy here does not
give the physical freedom present in unrestricted dense D=2 tensors.

### D=3 and D=4 representations

The selected D=3 state retains 0 plus 1/2 on each virtual leg, with actual
dimension and B unfolding rank three. A random complex D=3 start at seed
1234 instead collapsed to the dimension-one/dimension-two dimer pattern
and E=-0.3544444444444445. Seed 2026 and a pure-integer-triplet alternative
also gave higher energies. Explicit singlet-pair/spinon starts and a small
coefficient/sign screening were checked as additional diagnostics; they
did not produce a lower measured energy than the selected real start.
Initial screening data are labelled separately from evolved states.

The selected D=4 state has two actual four-dimensional leg types:
**1/2 plus 1/2** and **0 plus 1**. Every B unfolding has full rank four.
Both the multiplicity of the half-spin and the additional integer-spin
fusion channel participate. A different D=4 start gives a higher-energy
candidate near -0.42851 in the chi8 screening; the seed-2026 start was also higher in the
initial energy screening. Thus a nominal D alone is insufficient to
describe either the selected representation or optimization quality.

## Evolution and convergence limits

The initial schedule was dt=[.03,.01,.003,.001,.0005], tau=[40,20,5,2,1].
Each stage can stop only after three consecutive 100-step comparisons meet
both the lambda-spectrum and normalized nine-spin-wavefunction tolerances
1e-8, with at least 300 steps. Original boundary half-steps are retained.
An additional **1000 full steps at dt=.0005** were always performed and
both the preceding and following states saved for infinite-energy checks.

D=2 and the selected D=3 each took 3000 full steps, tau=18.3, including the
extra hold. Their final lambda and finite-wavefunction drifts were zero
or below 7e-14. Infinite energies before and after the hold agree to
rounding at chi16 for the selected D=3 state.

D=4 did not pass those local stopping tolerances at the first schedule's
cap. It was continued with maximum tau=[400,40,10,5,2] at the same dt list,
then held for another 1000 fine steps. Including both holds, the selected
state underwent **35933 full steps, tau=383.99**, excluding initialization.
At the same chi33, its energies before/after the last hold are
-0.4336944238793308 and -0.4336944413459763: change **-1.7467e-8/spin**.
Site magnetizations change by about 1e-6 over this hold. The energy is
stable at the reported precision, but final lambda drift 1.38e-7 and
finite-wavefunction drift 9.19e-7 remain above the stricter 1e-8 local
stopping target. This is not a claim of complete tensor convergence.

Environment checks:

| State | chi | E/spin | CTM converged? |
|---|---:|---:|---|
| D3 final | 16 | -0.3642946641698455 | yes |
| D3 final | 24 | -0.3642942150160775 | yes |
| D3 final | 33 | -0.3642942148193873 | yes |
| D4 final | 8 | -0.4336936696485819 | yes, dense diagnostic |
| D4 final | 24 | -0.4336944396430338 | **no**, 100 iterations, error 2.492e-6 |
| D4 final | 25 | -0.4336944396747932 | yes |
| D4 final | 33 | -0.4336944413459763 | yes |

The chi25-to-33 change is 1.67e-9. Actual D4 chi33 corner dimensions range
23..29; the cap was genuinely increased beyond the chi25 result. Some
chi16 and dense diagnostic environments stagnated and were stopped;
their logs are retained and they are not treated as converged results.

## Comparison and interpretation

Earlier unrestricted dense SU candidates at the same h and cell include
D2 E=-0.4128508298596544 (chi40) and D4 E=-0.437374106486 (chi24).
The present SU(2) candidates are higher by about 0.0584064 and 0.00367967.
The older random-origin D3 SU result, E=-0.42766761617027, has m=0.1208885,
so it is not in the same exactly selected magnetization sector.

These differences do **not** establish an energy limit imposed by SU(2).
The current charged SU update uses the norm of the complete auxiliary
multiplet in its local environment. After projection in an infinite array,
that environment is not the variational environment of the polarized
physical state. In particular, the poor D=3 SU fixed point must not be
reported as an optimized D=3 SU(2) ground-state energy. Gradient optimization
of the projected infinite-system energy has not been performed here.

## Reproduction and saved data

Edit **run_su2_D234.jl** directly. It contains the cases, parity matrix,
seeds, field, thread count, schedule, stopping thresholds and hold settings.
Set `continuation_input` to an existing after_hold/state.jld2 to use its
separate continuation schedule. No command-line options are needed.

The original random D3-to-D2 procedure failed at seed 2026 because separately
chosen HOSVD spaces produced an empty compatible channel. The new initializer
uses an explicit compatible parity pattern instead. This changes initialization,
not the original SU truncation kernels. New initialization/rank tests pass
**7/7 in TensorKit 0.14.11 and 7/7 in 0.17.0**. A further 20 comparisons
checked that the convergence driver's fixed-length evolution matches the
original update sequence and preserves the selected charge.

Local data root: `tmp_kagome_su2_cells_20261028/validation/` under the Python
workspace (used only as a working-data directory). All measurements and
flags: `su2_D234_all_measurements.csv`; selected final rows:
`su2_D234_selected_results.csv`; representation/rank details:
`selected_su2_bond_ranks.toml`. The corresponding directories retain SU2
checkpoints, U1 checkpoints, portable U1/dense files, finite-torus reports,
CTM JSON and logs. Tensors and large logs are not committed to Git.
