include(joinpath(@__DIR__,"../../../../src/fermionic/kagome/spin/bootstrap.jl"))
length(ARGS)==2 || error("Usage: run.jl {su|fu|ad} config.toml")
KagomeSpinOptimization.run_spin_optimization(ARGS...)
