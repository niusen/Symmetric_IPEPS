# One D4/chi40 measurement, CPU, no SU. All settings are here.
import Pkg
const BENCH_CELL_ROOT=normpath(joinpath(@__DIR__,".."))
if isnothing(Base.active_project()) || normpath(Base.active_project())!=joinpath(BENCH_CELL_ROOT,"Project.toml")
    Pkg.activate(BENCH_CELL_ROOT)
end
using Dates
benchmark_settings=(
    dims=[4],
    checkpoints=Dict(4=>joinpath(BENCH_CELL_ROOT,"results","u1_from_D4_D4_D10_h038_chi40","D4","final.jld2")),
    chis=[40],
    sizes=[3,6,9],
    contraction_size=15,
    out=joinpath(BENCH_CELL_ROOT,"results","variance_benchmark_D4_chi40_"*Dates.format(now(),dateformat"yyyymmdd_HHMMSS")),
    threads=10,ctm_iterations=50,ctm_tolerance=1e-8,sweeps=2,source_shift=1.,
)
bench_module=joinpath(BENCH_CELL_ROOT,"..","..","..","src","bosonic","kagome",
    "native_measurement","energy_variance","KagomeEnergyVariance.jl")
bench_clock=time_ns()
include(bench_module)
using .KagomeEnergyVariance
bench_load_seconds=Float64(time_ns()-bench_clock)/1e9
run_variance_benchmark(benchmark_settings;startup_seconds=bench_load_seconds)
