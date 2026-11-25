include(joinpath(@__DIR__,"run_optimization.jl"))
if abspath(PROGRAM_FILE)==@__FILE__
    run_optimization(:full_update,isempty(ARGS) ? joinpath(@__DIR__,"full_update.toml") : only(ARGS))
end
