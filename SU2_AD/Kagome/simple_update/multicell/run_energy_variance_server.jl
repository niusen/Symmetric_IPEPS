# Submit with: sh run_energy_variance_server.sh
# All physics/batch settings are here. Saved U1 states are READ ONLY; no SU.
variance_dims=collect(4:15)
variance_Ls=[3,6,8,12,15]           # Summation windows; need not be multiples of the state period.
variance_chi=40                    # One fixed chi for every saved D.
variance_settings=(
    dims=variance_dims,
    checkpoints=Dict(D=>joinpath(@__DIR__,"results",
        D<=10 ? "u1_from_D4_D4_D10_h038_chi40" : "u1_from_D10_D11_D15_h038_chi40",
        "D$D","final.jld2") for D in variance_dims),
    chis=[variance_chi],
    sizes=variance_Ls,               # Extract these L-by-L sums from ONE contraction.
    # Automatic padding for 2x2 observable patches and the 3 representative centers.
    # This is not a second independently scanned truncation parameter.
    contraction_size=3*cld(maximum(variance_Ls)+6,3), # only this grid must tile the period-3 state
    out=joinpath(@__DIR__,"results","variance_D4_D15_chi40_L3_6_8_12_15_Lmax21_right_warmup"),
    threads=10,
    ctm_iterations=50,
    ctm_tolerance=1e-8,
    sweeps=2,
    extra_bottom_warmup=false,      # Default: right warmup only; ordinary bottom scans remain.
    source_shift=1.0,
)

const VARIANCE_LIBRARY=normpath(joinpath(@__DIR__,"..","..","..","src","bosonic",
    "kagome","native_measurement","energy_variance","KagomeEnergyVariance.jl"))
if !isdefined(@__MODULE__,:KagomeVariancePlan)
    include(joinpath(dirname(VARIANCE_LIBRARY),"variance_batch_plan.jl"))
end
if !isdefined(@__MODULE__,:KAGOME_VARIANCE_SETTINGS_ONLY)
    import Pkg
    if isnothing(Base.active_project()) || normpath(Base.active_project())!=joinpath(@__DIR__,"Project.toml")
        Pkg.activate(@__DIR__)
    end
    module_clock=time_ns()
    include(VARIANCE_LIBRARY)
    using .KagomeEnergyVariance
    module_seconds=Float64(time_ns()-module_clock)/1e9
    run_variance_batch(variance_settings;startup_seconds=module_seconds)
end
