using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using MPSKitModels, LinearAlgebra, OptimKit
using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives


cd(@__DIR__)
include("..\\resource_codes\\kagome_load_tensor.jl")
#include("..\\resource_codes\\kagome_CTMRG.jl")
include("kagome_CTMRG.jl")
include("..\\resource_codes\\kagome_model.jl")
include("..\\resource_codes\\kagome_IPESS.jl")
include("..\\resource_codes\\kagome_FiniteDiff.jl")
include("..\\resource_codes\\Settings.jl")



Random.seed!(12345)


D=3;
chi=40;


theta=0*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);





Bond_irrep="A";
Triangle_irrep="A1+iA2";
nonchiral="A1_even";#"No", "A1_even"





ctm_setting=CTMRG_settings();
ctm_setting.CTM_conv_tol=1e-6;
ctm_setting.CTM_ite_nums=50;
ctm_setting.CTM_trun_tol=1e-8;
ctm_setting.svd_lanczos_tol=1e-8;
ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
ctm_setting.conv_check="singular_value";
ctm_setting.CTM_ite_info=true;
ctm_setting.CTM_conv_info=true;
ctm_setting.CTM_trun_svd=false;
ctm_setting.construct_double_layer=true;

dump(ctm_setting);


optim_setting=Optim_settings();
optim_setting.init_statenm="nothing";#"LS_A1even_U1_D_6_chi_60.json";#"nothing";
optim_setting.init_noise=0;
optim_setting.grad_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"

dump(optim_setting);

energy_setting=Energy_settings()
energy_setting.kagome_method ="E_single_triangle";
energy_setting.E_up_method = "1x1";
energy_setting.cal_chiral_order = false;

dump(energy_setting);

A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
global A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb  
#run_FiniteDiff(parameters,D,chi,Bond_irrep,Triangle_irrep,nonchiral,ctm_setting,optim_setting,energy_setting);


json_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd=initial_state(Bond_irrep,Triangle_irrep,nonchiral,D,"nothing",0)
bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb)

PEPS_tensor=bond_tensor;
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

H_triangle, H_Heisenberg, H12_tensorkit, H31_tensorkit, H23_tensorkit =Hamiltonians(U_phy,J1,J2,J3,Jchi,Jtrip)

AA_H,_=build_double_layer(A_fused,H_triangle);
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(10,A_fused,"PBC",false,true);


function ob(CTM,AA_fused)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA_fused[2,5,-2,3]*Tset.T3[-3,5,4];
    Norm= @tensor envL[1,2,3]*envR[1,2,3];
    Norm=real(Norm);


    return Norm;
end
ob(CTM,AA_H)

function cfun(A_fused, CTM)
    
    
    function fun(A_fused)
        CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM_test(10,A_fused,"PBC",false,true);
        AA_H,_=build_double_layer(A_fused,H_triangle);
        AA,_=build_double_layer(A_fused,[]);
        E=ob(CTM,AA_H)/ob(CTM,AA_fused)
        println(E)
        return E
    end

    ∂E = fun'(A_fused)

    CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM_test(10,A_fused,"PBC",false,true);
    AA_H,_=build_double_layer(A_fused,H_triangle);
    AA,_=build_double_layer(A_fused,[]);
    E=ob(CTM,AA_H)/ob(CTM,AA_fused)
    

    @assert !isnan(norm(∂E))
    
    return E,∂E
end

a,b=cfun(A_fused,CTM)








# function cfun(x)
#     (ψ,env) = x

#     function fun(peps)
#         env = leading_boundary(peps, alg_ctm, env)
#         x = H_expectation_value(peps, env, H)   
#         return x
#     end

#     ∂E = fun'(ψ)
#     env = leading_boundary(ψ, alg_ctm, env)
#     E = H_expectation_value(ψ, env, H)

#     @assert !isnan(norm(∂E))
#     return E,∂E
# end

# # my_retract is not an in place function which should not change x
# function my_retract(x,dx,α::Number)
#     (ϕ,env0) = x
#     ψ = deepcopy(ϕ)
#     env = deepcopy(env0)
#     ψ.A .+= dx.A .* α
#     #env = leading_boundary(ψ, alg_ctm,env)
#     return (ψ,env),dx
# end

# my_inner(x,dx1,dx2) = real(dot(dx1,dx2))

# function my_add!(Y, X, a)
#     Y.A .+= X.A .* a
#     return Y
# end

# function my_scale!(η, β)
#     rmul!(η.A, β)
#     return η
# end


# function init_psi(d::Int, D::Int, Lx::Int, Ly::Int)
#     Pspaces = fill(ℂ^d,Lx,Ly)
#     Nspaces = fill(ℂ^D,Lx,Ly)
#     Espaces = fill(ℂ^D,Lx,Ly)

#     Sspaces = adjoint.(circshift(Nspaces, (1, 0)))
#     Wspaces = adjoint.(circshift(Espaces, (0, -1)))

#     A = map(Pspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
#         return TensorMap(rand, ComplexF64, P ← N * E * S * W)
#     end

#     return InfinitePEPS(A)
# end


# alg_ctm = CTMRG(
#             verbose=10000,
#             tol=1e-10,
#             trscheme=truncdim(10),
#             miniter=4,
#             maxiter=200
#         )

# function main(;d=2,D=2,Lx=1,Ly=1)
#     ψ = init_psi(d,D,Lx,Ly)   
#     env = leading_boundary(ψ, alg_ctm) 
#     optimize(
#         cfun, 
#         (ψ,env),
#         ConjugateGradient(verbosity=3); 
#         inner=my_inner,
#         retract=my_retract,
#         scale! = my_scale!,
#         add! = my_add!
#     )
#     return ψ
# end

# main()
