using Revise, TensorKit
using LinearAlgebra, OptimKit
using TensorKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
using Random
using LineSearches
using Dates
cd(@__DIR__)

include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")

Random.seed!(888)




filenm="stochastic_iPESS_LS_D_8_chi_40_3.4728.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
data=load(filenm);
state=data["x"];
Lx,Ly=size(state)


B_set=Matrix{TensorMap}(undef,Lx,Ly);
T_set=Matrix{TensorMap}(undef,Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        B_set[ca,cb]=state[ca,cb].Tm;
        T_set[ca,cb]=state[ca,cb].Bm;
        # state_new[ca,cb]=Triangle_iPESS(Tset[ca,cb],Bset[ca,cb]);
        # iPESS_to_iPEPS(state_new[ca,cb]);
    end
end



save_filenm="B_T_sets_"*filenm;
jldsave(save_filenm;B_set,T_set);

