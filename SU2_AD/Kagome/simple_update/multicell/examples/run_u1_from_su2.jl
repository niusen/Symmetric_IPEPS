# All options are here. Run this file directly in Julia; no command-line flags.
using Dates
u1_from_su2_settings = (
    initial_state = joinpath(@__DIR__, "..", "data", "su2_D4_seed.toml"),
    Dmax = 4,
    component = 1,
    h = 0.38,                         # H = H_exchange + h*sum(Sz)
    threads = 10,
    chi = 16,                         # Local trial; final energy also checked at chi=24.
    ctm_iterations = 50,
    ctm_tolerance = 1e-6,
    truncation_tolerance = 1e-6,
    # Short local trial: 200+200+200=600 full Trotter steps, not the server scan.
    schedule = [(dt=0.01, tau=2.0), (dt=0.003, tau=0.6), (dt=0.001, tau=0.2)],
    out = joinpath(@__DIR__, "..", "results", "u1_from_su2_D4_"*Dates.format(now(),"yyyymmdd_HHMMSS")),
)

if !isdefined(@__MODULE__, :KAGOME_SETTINGS_ONLY)
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    if !isdefined(@__MODULE__, :run_u1_from_su2)
        include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "kagome", "workflows", "u1_from_su2_workflow.jl"))
    end
    run_u1_from_su2(u1_from_su2_settings)
end
