# All settings are here. Submit once with sh run_u1_D4_D10_server.sh.
u1_server_settings = (
    dims = collect(4:10),
    h = 0.38,                         # H = H_exchange + h*sum(Sz)
    chi = 40,
    threads = 10,
    device = "cpu",
    ctm_iterations = 50,
    ctm_tolerance = 1e-6,
    schedule = [(dt=0.03, tau=40.0), (dt=0.01, tau=20.0),
                (dt=0.003, tau=5.0), (dt=0.001, tau=2.0), (dt=0.0005, tau=1.0)],
    truncation_tolerance = 1e-6,
    checkpoint_every = 250,           # SU state only; no intermediate CTMRG.
    progress_every = 100,
    initial_state = joinpath(@__DIR__, "data", "u1_D4_optimized_seed.toml"),
    dense_reference = joinpath(@__DIR__, "data", "dense_reference_h038.csv"),
    out = joinpath(@__DIR__, "results", "u1_from_D4_D4_D10_h038_chi40"),
)
# D is the total number of U1 charge states per bond, not the number of sectors.
# Sequential D -> D+1; each D gets five stages / 9000 full Trotter steps.
# Fixed 1/9 sector. One final measurement per D, including all local observables.
if !isdefined(@__MODULE__, :KAGOME_SETTINGS_ONLY)
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    if !isdefined(@__MODULE__, :run_u1_server)
        include(joinpath(@__DIR__, "..", "..", "..", "src", "bosonic", "kagome", "native_measurement", "u1_server_workflow.jl"))
    end
    run_u1_server(u1_server_settings)
end
