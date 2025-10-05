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


include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate_iPESS.jl")

include("..\\..\\src\\bosonic\\symmetry_lib.jl")
# include("..\\..\\src\\bosonic\\save_jason_data.jl")

filenm="extended_Lx8_Ly2_D_8.jld2";
data=load(filenm);

if haskey(data,"T_set")
    T_set=data["T_set"];
    B_set=data["B_set"];
else
    state=data["x"];
    Lx,Ly=size(state);
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
end

for cx=1:size(B_set,1)
    for cy=1:size(B_set,2)
        B_set[cx,cy]=convert_SU2_to_Z2(B_set[cx,cy]);
        T_set[cx,cy]=convert_SU2_to_Z2(T_set[cx,cy]);
    end
end

jldsave("Z2_"*filenm;B_set,T_set);