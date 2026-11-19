# Small fixed-state numerical diagnostic; no state evolution or optimization.
# All options are here. Run under the multicell Project.toml environment.
root=normpath(joinpath(@__DIR__,"..","..","..","..",".."))
native=joinpath(root,"src","bosonic","kagome","native_measurement")
ENV["KAGOME_VARIANCE_NATIVE_ROOT"]=native
include(joinpath(native,"energy_variance","variance_warmup_comparison.jl"))
using .KagomeWarmupComparison

comparison_settings=(
    out=joinpath(@__DIR__,"validation_warmup_D2"), # must be a fresh directory
    chis=[4,8],
    sweep_counts=[1,2],
    seed=701,
    threads=10,
)
run_comparison(;comparison_settings...)
