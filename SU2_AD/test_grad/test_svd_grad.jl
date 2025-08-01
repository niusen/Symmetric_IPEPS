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
# include("../src/fermionic/Fermionic_CTMRG_unitcell_iPESS.jl")
include("../src/fermionic/square_Hubbard_model_cell.jl")
include("../src/fermionic/swap_funs.jl")
include("../src/fermionic/mpo_mps_funs.jl")
include("../src/fermionic/double_layer_funs.jl")
include("../src/fermionic/square_Hubbard_AD_cell.jl")
include("../src/fermionic/triangle_fiPESS_method.jl")

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);

global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=1e-8;



data=load("stochastic_iPESS_LS_D_8_chi_80_2.3399.jld2");




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


chi=30;
function cfun1(M)
    M=permute(M,(1,2,),(3,4,));
    uM,sM,vM = tsvd(M; trunc=truncdim(chi));
    sM_inv_sqrt=sdiag_inv_sqrt(sM);

    mm=uM*sM_inv_sqrt*vM;
    return real(dot(mm,mm))
end

y1=cfun1(AA);

∂E1=gradient(x ->cfun1(x), AA)[1];


function cfun2(M)
    M=permute(M,(1,2,),(3,4,));
    uM,sM,vM = tsvd(M; trunc=truncdim(chi));
    sM_inv_sqrt=sdiag_inv_sqrt(sM);

    mm=uM*sM_inv_sqrt*vM;
    return real(dot(mm,mm))
end

y2=cfun2(AA);

∂E2=gradient(x ->cfun2(x), AA)[1];

dot(∂E1,∂E2)/sqrt(dot(∂E1,∂E1)*dot(∂E2,∂E2))

#check multiplet truncation

M=permute(AA,(1,2,),(3,4,));
uM0,sM0,vM0 = tsvd(M; trunc=truncdim(chi+10));
uM1,sM1,vM1 = tsvd(M; trunc=truncdim(chi));
uM2,sM2,vM2 = tsvd(M; trunc=truncdim(chi; multiplet_tol=0.018));
println(space(sM2))
for (a,b) in blocks(sM0)
    println(a)
    if a==Irrep[SU₂](0)
        println(a,diag(b)[1:16])
    end
end


data=load("test.jld2");
M=data["M"];
chi_=data["chi_"];
uM_,sM_,vM_ = tsvd(M; trunc=truncdim(chi_));
println(space(sM_))
uM2_,sM2_,vM2_ = tsvd(M; trunc=truncdim(chi_; multiplet_tol=1e-5));
println(space(sM2_))
uM2_,sM2_,vM2_ = tsvd(M; trunc=truncdim(chi_+3; multiplet_tol=1e-5));
println(space(sM2_))
uM_,sM_,vM_ = tsvd(M);
for (a,b) in blocks(sM2_)
    println(a)
    if a==Irrep[SU₂](1/2)
        println(a,diag(b)[:])
    end
end