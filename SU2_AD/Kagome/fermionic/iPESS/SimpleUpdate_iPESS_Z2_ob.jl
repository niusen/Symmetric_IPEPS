include(joinpath(@__DIR__, "../../../src/fermionic/kagome/workflows/simple_update.jl"))
main(config_path=joinpath(@__DIR__, "config.toml")) = run_simple_update(config_path)
if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? joinpath(@__DIR__, "config.toml") : only(ARGS))
end
