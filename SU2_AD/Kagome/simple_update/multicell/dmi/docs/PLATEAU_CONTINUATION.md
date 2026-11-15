# Two Julia plateau-seed tasks

Both tasks use pure Julia TensorKit, the nine-spin sqrt3 cell, unrestricted dense
tensors, CPU/10 threads, chi=12, CTMRG at most 50 iterations and tol=1e-6.
All options are in the corresponding .jl file. No Python or plotting is called.

## Starting state

The default seed is the reliable, low-energy D=2, Dz=0.02, h=0.32, trial=1 result
from the latest independent-random run:

```text
multicell/results/DMI_independent_random3_D2_D4_20261114/
  D2_Dz0.02_random1/h0.32/final.jld2
  D2_Dz0.02_random1/h0.32/independent_result.toml
```

It has M=m/m_sat=0.11111111109567016 and e=-0.4097339175808129 at h=0.32.
Both files must already exist on the server. The job checks D, J, Dz, physical h,
magnetization, measurement reliability, dense geometry and the content hashes.
There is no fallback to a random or older plateau state. Raw dense JLD2 files
require a compatible TensorKit environment; use the environment that created it.

## Task 1: increase D at fixed Dz=0.02

From **dmi/server/** run `sh run_platform_D_chain.sh`.

Default D=2,3,...,10 and h=0.28,0.32,0.36,0.38,0.40,0.44,0.48: 63 points.
Each of the seven D=2 runs starts from the SAME original plateau seed. For every
later D, the initial ket is the preceding D's final ket at the SAME h.
There is no field continuation. Native SU/HOSVD grows Schmidt channels with the
new Dmax, preserving the loaded ket before evolution. No random padding/noise is
introduced. The actual retained dimensions, not just the cap, are recorded.

Output: `results/DMI_platform_D2_D10_Dz002_sqrt3_chi12/`.

## Task 2: increase positive/negative Dz at fixed D=2

From **dmi/server/** run `sh run_platform_Dz_signed.sh`.

Default Dz=0 and both signs of 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70. The same seven fields
give 33 Dz values and 231 points. Each h first evolves the SAME original seed at
Dz=+0.02. Then two separate paths leave that base:

```text
+0.02 -> +0.03 -> +0.05 -> +0.075 -> +0.10 -> +0.15 -> ... -> +0.70
+0.02 -> +0.01 -> 0 -> -0.01 -> -0.02 -> -0.03 -> ... -> -0.70
```

Every Dz uses its preceding Dz at the SAME h. The decreasing path does not pass
through the high-positive-Dz state, and no conjugation/reset is silently applied
to the seed. h>0 always means H_field=-h Sz; only Dz changes sign.

Output: `results/DMI_platform_D2_signed_Dz070_sqrt3_chi12/`.

The expanded task writes a new Dz070 output directory and does not overwrite
the earlier Dz010 data. Its h window remains 0.28--0.48; a plateau shifted outside
this window is not excluded by its absence in the scan. The grid samples 0.25,
0.28 and 0.30, near the spinon band-inversion scale discussed in Kumar et al.;
that band-inversion scale is not a magnetic-plateau boundary.

## Evolution, data and restart

Every point uses dt=(.03,.01,.003,.001,.0005), tau=(40,20,5,2,1), approximately
9000 full steps. These are fixed evolution lengths, not a proof of SU convergence.
There is one final CTMRG/measurement per point; no intermediate energy measurements.
The old SU/Trotter/HOSVD and factorized CTMRG functions are reused unchanged.

Each point saves its exact initial ket, stage/final checkpoints, parent path and
hash, complete observation TOML/JSON, site_magnetization.csv, bond_energies.csv,
triangle_observables.csv and continuation_result.toml/.json. Triangle field terms
receive half the local field, preserving the original counting convention.
The 3x3 coordinate display repeats three independent PESS, i.e. nine physical
spins; independent_cell labels make this explicit in site/bond tables.

The root contains settings.toml, plan.json (including parent indices), progress.csv
and COMPLETED.txt. SU time and measurement time are recorded separately. Raw
measurement failures/unconverged CTMRG stay flagged; chi is not silently enlarged.
Re-submit the same sh to skip completed points and resume completed SU stages.
An interrupted stage repeats from its saved start. Source/settings changes require
a new output directory. The two tasks have separate locks/output folders.

## Interpretation

These tasks track a known plateau branch. Exact imaginary-time evolution preserves
total Sz, so an exactly fixed-magnetization input may remain locked even without
an explicit U1 tensor constraint. A flat M alone is NOT evidence for the ground-state
plateau. Compare the full energies against the independently initialized states.
No perturbation has been added to the requested seed.

For this JSS + Dz*kappa - hSz Hamiltonian, complex conjugation maps Dz to -Dz
without changing Sz. The corresponding exact spectra and ground-state magnetization
curves agree at opposite signs. Asymmetry between these continuation paths can
therefore diagnose different metastable branches/truncation/optimization effects.

## Upload

Update `src/bosonic/kagome/dmi/plateau_continuation.jl`, both new .jl and .sh files
under **dmi/server/**, and optionally this document and the new small test under
**dmi/tests/**. Keep the existing src dependencies, Project.toml and Manifest.toml.
The seed data above must remain in the server's results directory. If it is already
there, no data upload is needed. Exclude **dmi/data/** when uploading source code.

The small test is `julia --project=. dmi/tests/test_plateau_continuation.jl`, run
from multicell. It uses a D=1 fixture and at most D=2, one-step evolutions only;
it never starts either production task.
