# All settings are here. Submit once with sh run_u1_D10_D15_server.sh.
# D10 is READ ONLY: load the saved final state, then evolve D11 -> ... -> D15.
u1_continuation_settings = (
    source_D = 10,
    dims = collect(11:15),
    h = 0.38,                         # Existing Julia convention: Hex + h*sum(Sz).
    chi = 40,
    threads = 10,
    device = "cpu",
    ctm_iterations = 50,
    ctm_tolerance = 1e-6,
    schedule = [(dt=0.03, tau=40.0), (dt=0.01, tau=20.0),
                (dt=0.003, tau=5.0), (dt=0.001, tau=2.0), (dt=0.0005, tau=1.0)],
    truncation_tolerance = 1e-6,
    checkpoint_every = 250,           # State checkpoints only; no intermediate CTMRG.
    progress_every = 100,
    initial_state = joinpath(@__DIR__, "results", "u1_from_D4_D4_D10_h038_chi40", "D10", "final.jld2"),
    initial_sha256 = "b16ca062dd869dfe2431e00d0a201e0571d7935fcc1e91fbf5707dffb6627607",
    dense_reference = "",             # Keep old output columns; no external comparison.
    out = joinpath(@__DIR__, "results", "u1_from_D10_D11_D15_h038_chi40"),
)
# D is the TOTAL U1 virtual dimension, not the number of charge sectors.
# Same five stages and round(Int,tau/dt) as the original job: 9000 full steps/D.
# Initial tensors and all lambdas are read without randomization or projection.
# Increasing Dmax lets the existing U1 SVD grow the virtual spaces during SU.
# This is a fixed evolution schedule; it does not certify SU convergence.
if !isdefined(@__MODULE__, :KAGOME_SETTINGS_ONLY)
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    if !isdefined(@__MODULE__, :run_u1_continuation)
        include(joinpath(@__DIR__, "..", "..", "..", "src", "bosonic", "kagome", "native_measurement", "u1_continuation_workflow.jl"))
    end
    run_u1_continuation(u1_continuation_settings)
end
