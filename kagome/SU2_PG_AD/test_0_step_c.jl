using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using LinearAlgebra, OptimKit
using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives


cd(@__DIR__)
include("resource_codes\\kagome_load_tensor.jl")
#include("..\\resource_codes\\kagome_CTMRG.jl")
include("resource_codes\\kagome_CTMRG.jl")
include("resource_codes\\kagome_model.jl")
include("resource_codes\\kagome_IPESS.jl")
include("resource_codes\\kagome_FiniteDiff.jl")
include("resource_codes\\Settings.jl")



Random.seed!(12345)


D=6;
chi=40;


theta=0.2*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);





Bond_irrep="A";
Triangle_irrep="A1+iA2";
nonchiral="No";#"No", "A1_even"
ipess_irrep=IPESS_IRREP(Bond_irrep, Triangle_irrep, nonchiral);




ctm_setting=CTMRG_settings();
ctm_setting.CTM_conv_tol=1e-6;
ctm_setting.CTM_ite_nums=0;
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
optim_setting.init_statenm="nothing";#"LS_A1even_D_6_chi_40.json";#"nothing";
optim_setting.init_noise=0;
optim_setting.grad_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
optim_setting.grad_checkpoint=true;

dump(optim_setting);

energy_setting=Energy_settings()
energy_setting.kagome_method ="E_single_triangle";#"E_single_triangle", "E_triangle"
energy_setting.E_up_method = "1x1";#"1x1", "2x2"
energy_setting.cal_chiral_order = false;

dump(energy_setting);

A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
global A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb  

#run_FiniteDiff(parameters,D,chi,Bond_irrep,Triangle_irrep,nonchiral,ctm_setting,optim_setting,energy_setting);


json_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd=initial_state(Bond_irrep,Triangle_irrep,nonchiral,D,optim_setting.init_statenm,optim_setting.init_noise)
elementary_tensors=Elementary_tensors(A_set,B_set,A1_set,A2_set,A1_has_odd,A2_has_odd);
bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb)

PEPS_tensor=bond_tensor;
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
A_unfused=PEPS_tensor;

U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

H_triangle, H_Heisenberg, H12_tensorkit, H31_tensorkit, H23_tensorkit =Hamiltonians(U_phy,J1,J2,J3,Jchi,Jtrip)

global H_triangle
  

function build_double_layer_no_svd(A,operator)
    #display(space(A))
    A=permute(A,(1,2,),(3,4,5));
    U_L=@ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    U_D=@ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    U_R=(U_L)';
    U_U=(U_D)';
    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    # uM,sM,vM=tsvd(A);
    # uM=uM*sM

    # uM=permute(uM,(1,2,3,),())
    # V=space(vM,1);
    # U=@ignore_derivatives unitary(fuse(V' ⊗ V), V' ⊗ V);
    # @tensor double_LD[:]:=uM'[-1,-2,1]*U'[1,-3,-4];
    # @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    # vM=permute(vM,(1,2,3,4,),());
    # if operator==[]
    #     @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    #     @tensor double_RU[:]:=vM'[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];
    # else
    #     @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    #     @tensor double_RU[:]:=vM'[3,-2,-4,1]*operator[2,1]*double_RU[-1,3,-3,-5,2];
    # end
    # #display(space(double_RU))

    # double_LD=permute(double_LD,(1,2,),(3,4,5,));
    # double_LD=U_L*double_LD;
    # double_LD=permute(double_LD,(2,3,),(1,4,));
    # double_LD=U_D*double_LD;
    # double_LD=permute(double_LD,(2,1,),(3,));
    # #display(space(double_LD))
    # double_RU=permute(double_RU,(1,4,5,),(2,3,));
    # double_RU=double_RU*U_R;
    # double_RU=permute(double_RU,(1,4,),(2,3,));
    # double_RU=double_RU*U_U;
    # double_LD=permute(double_LD,(1,2,),(3,));
    # double_RU=permute(double_RU,(1,),(2,3,));
    # AA_fused=double_LD*double_RU;

    A=permute(A,(1,2,3,4,5,));
    if operator==[]
        @tensor AA_fused[:]:=A'[2,4,6,8,1]*A[3,5,7,9,1]*U_L[-1,2,3]*U_D[-2,4,5]*U_R[6,7,-3]*U_U[8,9,-4];
    else
        @tensor Ap[:]:=A'[-1,-2,-3,-4,1]*operator[-5,1];
        @tensor AA_fused[:]:=Ap[2,4,6,8,1]*A[3,5,7,9,1]*U_L[-1,2,3]*U_D[-2,4,5]*U_R[6,7,-3]*U_U[8,9,-4];
    end
    

    
    return AA_fused, U_L,U_D,U_R,U_U
end


function init_CTM_double_layer(A)
    U_L=@ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    U_D=@ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    U_R=(U_L)';
    U_U=(U_D)';
    @tensor C1[:]:=A'[2,4,6,3,1]*A[2,5,7,3,1]*U_D[-1,4,5]*U_R[6,7,-2];
    @tensor C2[:]:=A'[4,6,3,2,1]*A[5,7,3,2,1]*U_L[-1,4,5]*U_D[-2,6,7];
    @tensor C3[:]:=A'[6,3,2,4,1]*A[7,3,2,5,1]*U_U[4,5,-1]*U_L[-2,6,7];
    @tensor C4[:]:=A'[2,3,6,4,1]*A[2,3,7,5,1]*U_R[6,7,-1]*U_U[4,5,-2];

    @tensor T4[:]:=A'[2,3,5,7,1]*A[2,4,6,8,1]*U_D[-1,3,4]*U_R[5,6,-2]*U_U[7,8,-3];
    @tensor T1[:]:=A'[3,5,7,2,1]*A[4,6,8,2,1]*U_L[-1,3,4]*U_D[-2,5,6]*U_R[7,8,-3];
    @tensor T2[:]:=A'[5,7,2,3,1]*A[6,8,2,4,1]*U_U[3,4,-1]*U_L[-2,5,6]*U_D[-3,7,8];
    @tensor T3[:]:=A'[7,2,3,5,1]*A[8,2,4,6,1]*U_R[3,4,-1]*U_U[5,6,-2]*U_L[-3,7,8];

    Cset=Cset_struc(C1,C2,C3,C4);
    Tset=Tset_struc(T1,T2,T3,T4);

    CTM=CTM_struc(Cset,Tset)
    return CTM

end

_, U_L,U_D,U_R,U_U=build_double_layer(A_fused,[]);
CTM=init_CTM_double_layer(A_fused)

CTM= init_CTM(chi,A_fused,"PBC",false); #somehow I can't include grad here
#global CTM

function fun(x)
    global chi, ipess_irrep, elementary_tensors, H_triangle
    #Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=vector_to_coes(elementary_tensors, ipess_irrep, x);

    bond_tensor,triangle_tensor=construct_su2_PG_IPESS_vec(x,elementary_tensors, ipess_irrep);

    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;

    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];
    #A_fused=A_fused/norm(A_fused);
    AA_fused, U_L,U_D,U_R,U_U=build_double_layer(A_fused,[]);
    #global CTM
    #CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true,true);
    #CTM=init_CTM_double_layer(A_fused)


    
    CTM= init_CTM(chi,A_fused,"PBC",false); #somehow I can't include grad here
    norm_1site=ob_1site_closed(CTM,[],AA_fused,[],true);
    AA_H, _,_,_,_=build_double_layer(A_fused,H_triangle);
    E_up=ob_1site_closed(CTM,[],AA_H,[],true);
    E_up=E_up/norm_1site;
    E=real(E_up*2)/3;



    println(E)
    # global E_tem, CTM_tem, AA_fused_tem
    # CTM_tem=deepcopy(CTM);
    # AA_fused_tem=deepcopy(AA_fused);
    # E_tem=deepcopy(E)
    return E
end


function cfun(state_vec)
    

    ∂E = fun'(state_vec)
    #E=fun(state_vec)
    
    global E_tem, CTM_tem, AA_fused_tem
    @assert !isnan(norm(∂E))
    
    return E_tem,∂E,CTM_tem, AA_fused_tem
end

global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=ctm_setting.CTM_trun_tol
global ipess_irrep, elementary_tensors
state_vec=coes_to_vector(Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, ipess_irrep)

state_vec=normalize_IPESS_SU2_PG_vec(elementary_tensors, ipess_irrep, state_vec);
E,∂E,CTM_tem, AA_fused_tem=cfun(state_vec);
println(E,∂E)



dt=0.000001

E0=fun(state_vec);

grad=Vector{Float64}(undef,0);

for cc=1:length(state_vec)
    state_vec_tem=deepcopy(state_vec);
    state_vec_tem[cc]=state_vec_tem[cc]+dt;
    grad=vcat(grad,(fun(state_vec_tem)-E0)/dt);

end
println(grad)




# init=initial_condition(init_type="PBC", reconstruct=true, has_AA_fused=false);

# CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,[],ctm_setting,optim_setting)
# E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting1.kagome_method);

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