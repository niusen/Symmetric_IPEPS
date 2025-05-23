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

include("..\\..\\..\\..\\..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\..\\..\\..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\..\\..\\..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\..\\..\\..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\..\\..\\..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\..\\..\\..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\..\\..\\..\\..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\..\\..\\..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\..\\..\\..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\..\\..\\..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\..\\..\\..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\..\\..\\..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")

Random.seed!(888)

filenm="stochastic_iPESS_2x1_LS_D_7_chi_80_2.38534.jld2"

#filenm="SU_iPESS_SU2_csl_D"*string(D)*".jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
data=load(filenm);
x=data["x"]
Lx,Ly=size(x);

state_new=Matrix{Triangle_iPESS}(undef,Lx,Ly);
coe=0.5;
for ca=1:Lx
    for cb=1:Ly
        bm=x[ca,cb].Bm;
        tm=x[ca,cb].Tm;
        
        Pg=Gutzwiller_SU2(coe);
        @tensor bm[:]:=bm[-1,1,-3,-4]*Pg[-2,1];
        bm=permute(bm,(1,),(2,3,4,));
        state_new[ca,cb]=Triangle_iPESS(bm,tm);
        iPESS_to_iPEPS(state_new[ca,cb]);
    end
end






save_filenm="Gutzwiller_"*filenm;
jldsave(save_filenm;x=state_new);

