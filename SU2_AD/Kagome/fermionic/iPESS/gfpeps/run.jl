# Usage: julia run.jl config.toml [--check]
const GF_SOURCE=normpath(joinpath(@__DIR__,"../../../../src/fermionic/kagome/gfpeps"))
if "--check" in ARGS
    include(joinpath(GF_SOURCE,"workflows/config.jl"))
    check_gfpeps_job(only(filter(!=("--check"),ARGS)))
else
    include(joinpath(GF_SOURCE,"workflows/run_config.jl"))
    run_gfpeps_config(only(ARGS))
end
