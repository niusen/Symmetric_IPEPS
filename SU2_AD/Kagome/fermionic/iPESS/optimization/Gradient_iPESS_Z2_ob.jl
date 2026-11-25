include(joinpath(@__DIR__,"run_optimization.jl"))
if abspath(PROGRAM_FILE)==@__FILE__
    run_optimization(:variational,isempty(ARGS) ? joinpath(@__DIR__,"variational.toml") : only(ARGS))
end
