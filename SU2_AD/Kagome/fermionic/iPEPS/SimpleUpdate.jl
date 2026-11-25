# Run from SU2_AD: julia Kagome/fermionic/iPEPS/SimpleUpdate.jl
using LinearAlgebra, MAT, JLD2
include(joinpath(@__DIR__,"../../../src/fermionic/kagome/ipeps/bootstrap.jl"))
using .KagomeIPEPSSU

# Edit parameters in Julia, or pass MODEL INPUT OUTPUT on the command line.
family = :ipeps
model = :spin                 # :spin, :orig, :mild, :zero or :hubbard
input = joinpath(@__DIR__,"../iPESS/results/gfpeps_validation/ipeps_D4/tensors.mat")
output = joinpath(@__DIR__,"../iPESS/results/ipeps_su/spin_D4")
if !isempty(ARGS)
    args=copy(ARGS)
    if first(args) in ("ipeps","ipess")
        family=Symbol(popfirst!(args))
    end
    length(args)==3 || error("Usage: SimpleUpdate.jl [ipeps|ipess] MODEL INPUT OUTPUT")
    model,input,output=Symbol(args[1]),args[2],args[3]
end
# Default to the seed dimension; replace with a number to truncate/grow explicitly.
D_max=if endswith(lowercase(input),".mat")
    matopen(f->2^Int(read(f,"m")),input)
else
    load(input,"metadata").Dmax
end
J,h=1.,0.5
t=(1.,1.)                    # magnitudes for physical spin (up,down)
mu=nothing                   # use the seed fields; or set (mu_up,mu_down)
onsite_U=0.                  # :hubbard uses U*n_up*n_down
schedule=[(0.01,0.01)]       # (tau,dt) per stage; short example, not convergence
chi=12; ctm_iterations=40
BLAS.set_num_threads(1)
run_native_su(family,input,output,model;schedule,Dmax=D_max,J,h,t,mu,onsite_U,chi,maxiter=ctm_iterations)
