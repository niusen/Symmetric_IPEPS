# Pure Julia dense iPESS: every (D, Dz, h, trial) starts from a new random ket.
# Physical Hamiltonian: H = J*sum(SS) + Dz*sum_CCW(kappa_z) - h*sum(Sz), J=1.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "..", "..", "..", "..", "..", "src", "bosonic", "kagome", "dmi", "independent_random_fields.jl"))

# Edit all production settings HERE; no command-line physics options are needed.
options = (
    dims = [6, 8],
    Dz = [0., .02, .03, .05, .10, .15, .20, .30, .40, .50, .60, .70],
    fields = [.28, .32, .36, .38, .40, .44, .48, .52, .56, .60, .64],
    trials = [1, 2, 3],
    seed_base = 202611170000,
    threads = 10,
    trun_tol = 1e-8,
    chi = 12,
    ctm_max = 50,
    ctm_tol = 1e-6,
    fresh = [(dt=.03, tau=40.), (dt=.01, tau=20.), (dt=.003, tau=5.),
             (dt=.001, tau=2.), (dt=.0005, tau=1.)],
    warm = [], # Unused: there is NO D, Dz, h, or trial continuation.
    out = joinpath(@__DIR__, "..", "..", "results",
        "DMI_independent_random3_D6_D8_positive_Dz070_h028_064_sqrt3_chi12"),
)

# TensorKit uses CPU. No U1/SU2/C3 constraints and no saved plateau seed are used.
# Keep existing positive-field handling: the native Julia API receives -h.
# Reuse existing CTMRG: failed primary measurements trigger its chi=24 recheck.
# A chi=24 recheck does NOT perform any additional SU evolution.
# Each fresh point has 9000 full Trotter steps and one primary final measurement.
# The default grid has 2 * 12 * 11 * 3 = 792 independent optimizations.
# This field window is a screening window; absence inside it does not exclude
# a narrow plateau between samples or a plateau outside the field window.

# Including the file for the small test only loads its settings, without running
# any production points. Normal invocation from the .sh enters this block.
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    count_points = length(options.dims) * length(options.Dz) * length(options.fields) * length(options.trials)
    count_steps = sum(round(Int, stage.tau / stage.dt) for stage in options.fresh)
    println("$count_points INDEPENDENT random points; D=$(options.dims); Dz=$(extrema(options.Dz)); $(length(options.trials)) trials per point")
    println("$count_steps full SU steps EACH; CPU $(options.threads) threads; chi=$(options.chi), CTMRG max=$(options.ctm_max), tol=$(options.ctm_tol)")
    println("Existing chi=24 diagnostic recheck on failed primary measurements; data only, no plotting.")
    println("Output: $(abspath(options.out))")
    flush(stdout)
    KagomeDMIIndependentFields.run(options)
end
