# D=6,8 independent random DMI scan (pure Julia dense iPESS)

From **dmi/server/** run:

```sh
sh run_independent_random_D6_D8.sh
```

This starts ONE background Julia process and prints its PID and log path. Logs
and PID records are saved in **dmi/data/logs/**. There is no plotting on the
server, Python call, Slurm job, watchdog or automatic relaunch.

## Settings

Edit **server/run_independent_random_D6_D8.jl**, not command-line arguments.

- D = 6,8; sqrt3 x sqrt3 cell, three independent PESS cells / nine physical spins.
- Dz/J = 0,.02,.03,.05,.10,.15,.20,.30,.40,.50,.60,.70.
- h/J = .28,.32,.36,.38,.40,.44,.48,.52,.56,.60,.64.
- Three independent complex Gaussian seeds at EVERY (D,Dz,h): **792 points**.
- J=1, CCW DMI arrows; physical field term is -h Sz. The existing native Julia
  +h API receives -h, exactly as in the completed random/continuation tasks.
- Dense tensors; NO U1/SU2/C3 restriction, projected seed, saved platform seed,
  noise-added continuation, or transfer between D, Dz, fields or trials.
- CPU / 10 threads; chi=12, CTMRG max50, tolerance=1e-6, SU truncation tol=1e-8.
- Five SU stages: dt=[.03,.01,.003,.001,.0005], tau=[40,20,5,2,1].
  This is **9000 full Trotter steps per fresh point**. No mid-evolution CTMRG.
- One primary final measurement per point. The existing independent-random
  workflow automatically remeasures at chi=24 if the primary measurement fails
  its validity or CTMRG-convergence check; this does not run more SU steps.

Only the nonnegative Dz side is simulated, as requested. The preceding signed
scan found nearly symmetric results. In the Sz product basis the specified
Hamiltonian obeys H(Dz,h)*=H(-Dz,h); conjugating a ket changes its chirality and
keeps its Sz. A separate negative-side random scan is not part of this task.

The field upper bound is increased from .48 to .64 because the 1/9 crossing
moves toward higher fields for larger Dz. The grid remains a preliminary
screen: it can miss a very narrow plateau or a plateau outside this window.
Flat magnetization and a converged environment alone do not certify a
lowest-energy plateau. Compare trial energies and saved competing states.

## Data and resume

The new output directory is:

```text
multicell/results/DMI_independent_random3_D6_D8_positive_Dz070_h028_064_sqrt3_chi12/
```

The parent can be named **u1_multicell** on the server; paths are relative to
the entry file, so its name does not affect the task. Existing D=2--4 random
or platform-continuation directories are not reused.

The existing src/bosonic/kagome/dmi/independent_random_fields.jl workflow saves
settings.toml, plan.json, progress.csv, and a final COMPLETED.txt. Every point
has initialization metadata / actual seed audit, initial.jld2, stage.jld2,
final.jld2, and JSON/TOML observations (including local cell, site and bond
measurements). Stage histories remain in the point files/checkpoint metadata.
There is no separate figure generation or ground-state phase classification.

Re-submit unchanged settings to resume: completed points are skipped, and an
unfinished point resumes its own saved stage. A live writer is protected by
the existing PID lock. Do not start multiple writers for the same output.
Changing the settings requires a NEW output directory.

The scan is large. The previous D=8 continuation measurements took about
15 minutes per point (median) at chi=12, so hundreds of independent points
may take several days or longer. Random states and conditional chi=24
rechecks can differ in cost. This is not a measured runtime for the new scan.

## Server upload

If the completed Julia independent-random task already ran on this server,
upload ONLY these two new files to the existing **dmi/server/** directory:

1. run_independent_random_D6_D8.jl
2. run_independent_random_D6_D8.sh

No new package dependency or algorithm source change is needed. The existing
Project.toml, Manifest.toml, and src/ tree must still be present. No saved
platform state is needed. For a new installation, retain the existing Julia
package layout and dependencies rather than uploading the two entry files
alone.

## Small laptop check

From the multicell project directory:

```sh
julia --project=. dmi/tests/test_random_D6_D8_task.jl
```

This checks the full grid and unique seeds without allocating any D=6/8
states, then runs two D=1 random initializations for ONE short SU step in a
temporary folder. It verifies distinct actual kets, recorded seeds, output
files, and resume behavior. It never executes the production scan.
