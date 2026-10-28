> Historical validation of the former Python bridge. The current implementation
> is pure Julia; see SU2_D4_D12_SERVER_VALIDATION.md for current checks.

# Dense conversion / measurement validation, 2026-10-28

Inputs are the six-step charged-SU trials from `CHARGED_VALIDATION.md`,
with one auxiliary spin-1/2 per sqrt3 cell, seed 1234, initial virtual
space 0+1/2 and Dmax=3 or 6. These states are not converged ground states.

Local versions: Julia 1.11.6, TensorKit 0.17.0, PyTorch 2.5.1. The dense
CTMRG runs used the laptop's only visible GPU, cuda:0 (RTX 4050), and
10 CPU threads. The reusable server entry point defaults to cuda:1.

## Independent finite dense measurement

The script first expands each local SU(2) TensorMap, explicitly selects the
highest-weight auxiliary component and removes that leg. It then contracts
ordinary local arrays. For uniform-D compatibility it zero-pads virtual
legs, checking the physical wavefunction before and after padding.
Both padding wavefunction errors were exactly 0.0 in this validation.

Energy and magnetization are independently evaluated with explicit Pauli
operators, rather than the reference's bit-basis expectation code.

| Input Dmax | Finite normalized m | Finite energy/spin, h=0.38 | Largest difference from SU(2) reference |
|---:|---:|---:|---:|
| 3 | 0.1111111111111111 | -0.3434100171781438 | 4.72e-16 |
| 6 | 0.1111111111111111 | -0.3517596831309943 | 5.42e-16 |

The maximum difference includes exchange energy, total field energy, site
Sz, normalized magnetization and total S^2. Total physical Sz=-1/2 and
S^2=3/4 are reproduced. The finite torus has nine physical spins.

## Infinite dense CTMRG measurement

The exported arrays are loaded using iPEPS_Z2's standard dense kagome
ansatz. Its existing factorized Q/R double layers and observables are used
without modifying those kernels. No SU evolution is performed.

Tolerance is 1e-6, with a cap of 100 CTMRG iterations. These are the two
environment-converged results:

| Input Dmax | chi | Iterations | Corner error | Normalized m | Energy/spin, h=0.38 |
|---:|---:|---:|---:|---:|---:|
| 3 | 16 | 18 | 5.70e-7 | 0.111111109186716 | -0.355072345470200 |
| 6 | 8 | 19 | 2.30e-7 | 0.111111056342596 | -0.363433941449774 |

The corresponding exchange energies per spin are -0.333961234724724 and
-0.342322840744680. The onsite field energies are -0.021111110745476 and
-0.021111100705093. Both total-energy reconstructions satisfy
`e_total = e_exchange + 0.38*mean(Sz)` to better than 1.2e-16; the field is
included only once. The exact fixed-charge field contribution would be
-0.38/18 per spin. Finite-chi / environment errors account for the small
observed deviations from m=1/9.

At Dmax=3, the chi=8 run did NOT converge in 100 iterations (corner error
2.37e-3). Its energy was -0.355072311038199, differing from chi=16 by
3.44e-8, but it is not treated as an environment-converged result.

At Dmax=6, the supplemental chi=12 run showed a persistent corner error
near 2.67e-2. It was stopped after iteration 36 to keep the laptop check
bounded. Its progress log was retained; no final chi=12 energy or
convergence claim is reported. **The Dmax=6 energy is therefore a chi=8
measurement, not a demonstrated chi-converged value.**

Equivalent positions of the computational 3x3 grid agree in site Sz to
5.05e-12 (Dmax=3, chi=16) and 1.46e-8 (Dmax=6, chi=8). Actual corner
dimensions range from 13 to 16 in the former and are 8 in the latter.

## Provenance and scope

Original charged checkpoint SHA256:

```text
Dmax=3: b0c38bbc9735c2ae52be4cafa655e51222947fcf58654afc0bf0fa2d0ce9cfb2
Dmax=6: fd87bd30ec8585b362770afff60c8d4eb6059d0bf783f1c4e8f35f1fd9c54361
```

The finite and infinite energies refer to different systems and are not
expected to coincide. The close magnetization verifies the explicit
projection, geometry, padding and dense measurement path. The selected
auxiliary boundary condition itself fixes the charge density, so these
checks do not establish a stable 1/9 plateau. CTMRG convergence also does
not imply convergence of the preceding six-step SU state.
