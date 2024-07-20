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

include("..\\..\\..\\..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\..\\..\\..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\..\\..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\..\\..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\..\\..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\..\\..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\..\\..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")

Random.seed!(888)

D=8;


filenm="FU_iPESS_LS_D_8_chi_100.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
data=load(filenm);
Tset=data["T_set"];
Bset=data["B_set"];
Lx,Ly=size(Tset);

state_new=Matrix{Triangle_iPESS}(undef,Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        state_new[ca,cb]=Triangle_iPESS(Tset[ca,cb],Bset[ca,cb]);
        iPESS_to_iPEPS(state_new[ca,cb]);
    end
end






save_filenm="iPESS_FU_D_"*string(D)*".jld2"
jldsave(save_filenm;x=state_new);

