# Compatibility entry point. Production functions live under src.
include(joinpath(@__DIR__, "../../../../../src/fermionic/kagome/gfpeps/bootstrap.jl"))
const ROOT=KAGOME_WORKSPACE
BLAS.set_num_threads(parse(Int,get(ENV,"KAGOME_BLAS_THREADS","4")))
include("session_runs.jl")
if abspath(PROGRAM_FILE)==@__FILE__
    D=parse(Int,ARGS[1]);chi=parse(Int,ARGS[2])
    cutoff=length(ARGS)>2 ? parse(Float64,ARGS[3]) : 1e-10
    maxiter=length(ARGS)>3 ? parse(Int,ARGS[4]) : 100
    family=length(ARGS)>4 ? ARGS[5] : "ipess"
    validate_saved(family,D,chi;cutoff,maxiter,save_environment=true)
end
