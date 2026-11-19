using TensorKit, LinearAlgebra, JLD2
const SOURCE = normpath(joinpath(@__DIR__,"..","..","..","src","fermionic","kagome"))
include(joinpath(SOURCE,"KagomeFermionicZ2.jl"))
using .KagomeFermionicZ2
include(joinpath(SOURCE,"ctmrg.jl"))
using .KagomeFermionicCTM

function main(args=ARGS)
    1 <= length(args) <= 3 || error("Usage: julia compute_ob.jl state.jld2 [chi=20] [maxiter=100]")
    state,parameters,_ = load_checkpoint(args[1])
    chi = length(args)>=2 ? parse(Int,args[2]) : 20
    maxiter = length(args)>=3 ? parse(Int,args[3]) : 100
    BLAS.set_num_threads(1)
    result = measure_ctmrg(state,parameters;chi,maxiter)
    @show result.observables.energy_per_site result.observables.density
    @show result.observables.double_occupancy result.converged result.error
    result.converged || @warn "CTMRG has not converged."
    output = splitext(abspath(args[1]))[1]*"_chi$(chi)_ob.jld2"
    jldsave(output;parameters,observables=result.observables,chi,iterations=result.iterations,
             error=result.error,converged=result.converged)
    println("Saved $output")
    return result
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
