# Fixed-state contraction check: no simple update or tensor optimization.
# All options are here. Run: sh run_u1_D9_D10_chi_check.sh
chi_check_settings = (
    # These are the FINAL checkpoints from the existing D4 -> D10 U1 job.
    # Keep both entries, or comment out one entry to measure only D=9 or D=10.
    sources = [
        (D=9, checkpoint=joinpath(@__DIR__, "results", "u1_from_D4_D4_D10_h038_chi40", "D9", "final.jld2")),
        (D=10, checkpoint=joinpath(@__DIR__, "results", "u1_from_D4_D4_D10_h038_chi40", "D10", "final.jld2")),
    ],
    chis = [20, 30, 40, 60, 80],
    h = 0.38,                      # H = sum_<ij> Si.Sj + h*sum_i Sz_i; J=1.
    threads = 10,
    device = "cpu",
    ctm_iterations = 50,
    ctm_tolerance = 1e-6,
    out = joinpath(@__DIR__, "results", "u1_D9_D10_fixed_state_chi_check"),
)
# Each chi starts a fresh CTMRG environment on the SAME input state.
# Bond 'energy' means <Si.Sj>, excluding the magnetic field term.
# Finished points are reused on restart; input hashes/settings must match.
if !isdefined(@__MODULE__, :KAGOME_SETTINGS_ONLY)
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    include(joinpath(@__DIR__, "..", "..", "..", "src", "bosonic", "kagome", "native_measurement", "chi_scan_workflow.jl"))
    KagomeChiScan.run_chi_scan(chi_check_settings)
end
