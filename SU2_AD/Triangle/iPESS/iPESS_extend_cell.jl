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
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")



filenm="D_8.jld2";

Lx_new=4;
Ly_new=2;

filenm_new="extended_Lx"*string(Lx_new)*"_Ly"*string(Ly_new)*"_"*filenm;

global Lx,Ly




data=load(filenm);
if haskey(data,"x")
    global Lx,Ly

    x=data["x"];
    Lx=size(x,1);
    Ly=size(x,2);
    @assert mod(Lx_new,Lx)==0;
    @assert mod(Ly_new,Ly)==0;
    x_new=Matrix{Triangle_iPESS}(undef,Lx_new,Ly_new);
    for cx=1:Lx_new
        for cy=1:Ly_new
            x_new[cx,cy]=x[mod1(cx,Lx),mod1(cy,Ly)];
        end
    end

    x=x_new;
    jldsave(filenm_new;x);
    
else
    global Lx,Ly
    T_set=data["T_set"];
    B_set=data["B_set"];
    Lx,Ly=size(T_set);
    @assert mod(Lx_new,Lx)==0;
    @assert mod(Ly_new,Ly)==0;
    B_set_new=Matrix{TensorMap}(undef,Lx_new,Ly_new);
    T_set_new=Matrix{TensorMap}(undef,Lx_new,Ly_new);
    for cx=1:Lx_new
        for cy=1:Ly_new
            B_set_new[cx,cy]=B_set[mod1(cx,Lx),mod1(cy,Ly)];
            T_set_new[cx,cy]=T_set[mod1(cx,Lx),mod1(cy,Ly)];
        end
    end

    B_set=B_set_new;
    T_set=T_set_new;
    jldsave(filenm_new;B_set,T_set);
end



