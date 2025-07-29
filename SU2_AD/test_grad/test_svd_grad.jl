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
include("../src/fermionic/Fermionic_CTMRG_unitcell_iPESS.jl")
include("../src/fermionic/square_Hubbard_model_cell.jl")
include("../src/fermionic/swap_funs.jl")
include("../src/fermionic/mpo_mps_funs.jl")
include("../src/fermionic/double_layer_funs.jl")
include("../src/fermionic/square_Hubbard_AD_cell.jl")
include("../src/fermionic/triangle_fiPESS_method.jl")

data=load("stochastic_iPESS_LS_D_8_chi_80.jld2");


x=data["x"]
Lx,Ly=size(x);

B_set=Matrix{TensorMap}(undef,Lx,Ly);
T_set=Matrix{TensorMap}(undef,Lx,Ly);
A_set=Matrix{TensorMap}(undef,Lx,Ly);



for ca=1:Lx
    for cb=1:Ly
        B_set[ca,cb]=x[ca,cb].Bm;
        T_set[ca,cb]=x[ca,cb].Tm;
        A=permute((x[ca,cb].Tm)*(x[ca,cb].Bm),(1,5,4,2,3,));#L,D,R,U,d,
        A=permute(A,(1,2,3,4,5,));
        A=A/norm(A);
        A_set[ca,cb]=A;
    end
end


pos=[1,1];
AA, U_L,U_D,U_R,U_U=build_double_layer_swap(A_set[pos[1],pos[2]]',A_set[pos[1],pos[2]]);


function cfun1(M)
    M=permute(M,(1,2,),(3,4,));
    uM,sM,vM = my_tsvd(M; trunc=truncdim(30));

    mm=uM*sM*vM;
    return real(dot(mm,mm))
end

y=cfun(uM,sM,vM);

∂E=gradient(x ->cfun(x), A_set);
