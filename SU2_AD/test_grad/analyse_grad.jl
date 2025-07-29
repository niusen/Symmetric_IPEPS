using Revise
using LinearAlgebra:diag,I,diagm 
using TensorKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
using Random
using LineSearches,OptimKit
using Dates
cd(@__DIR__)

include("../src/bosonic/Settings.jl")
include("../src/bosonic/Settings_cell.jl")
include("../src/bosonic/iPEPS_ansatz.jl")
include("../src/bosonic/AD_lib.jl")
include("../src/bosonic/line_search_lib.jl")
include("../src/bosonic/line_search_lib_cell.jl")
include("../src/bosonic/stochastic_opt.jl")
include("../src/bosonic/optimkit_lib.jl")
include("../src/bosonic/CTMRG.jl")
include("../src/fermionic/Fermionic_CTMRG.jl")
include("../src/fermionic/Fermionic_CTMRG_unitcell.jl")
include("../src/fermionic/square_Hubbard_model_cell.jl")
include("../src/fermionic/swap_funs.jl")
include("../src/fermionic/mpo_mps_funs.jl")
include("../src/fermionic/double_layer_funs.jl")
include("../src/fermionic/square_Hubbard_AD_cell.jl")
include("../src/fermionic/triangle_fiPESS_method.jl")
Random.seed!(888)
println("stochastic opt")


data=load("test_grad.jld2");
Bset=Matrix{TensorMap}(undef,2,2);
Tset=Matrix{TensorMap}(undef,2,2);
for ca=1:2
    for cb=1:2
        Bset[ca,cb]=unthunk(data["∂E"][ca,cb].Bm)
        Tset[ca,cb]=unthunk(data["∂E"][ca,cb].Tm)
    end
end

@show norm.(Bset)
@show norm.(Tset)