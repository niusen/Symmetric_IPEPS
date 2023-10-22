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
include("..\\resource_codes\\kagome_CTMRG.jl")
include("resource_codes\\kagome_model.jl")
include("resource_codes\\kagome_IPESS.jl")
include("resource_codes\\kagome_FiniteDiff.jl")
include("resource_codes\\Settings.jl")
include("resource_codes\\AD_lib.jl")



Random.seed!(12345)


D=8;
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
ctm_setting.CTM_ite_nums=10;
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
optim_setting.init_statenm="julia_LS_D_8_chi_40.json";#"LS_A1even_D_6_chi_40.json";#"nothing";
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


@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
A_unfused=PEPS_tensor;

U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

H_triangle, H_Heisenberg, H12_tensorkit, H31_tensorkit, H23_tensorkit =Hamiltonians(U_phy,J1,J2,J3,Jchi,Jtrip)



function fun(x)
    global chi, ipess_irrep, elementary_tensors
    #Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=vector_to_coes(elementary_tensors, ipess_irrep, x);

    bond_tensor,triangle_tensor=construct_su2_PG_IPESS_vec(x,elementary_tensors, ipess_irrep);
    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;

    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

    # norm_A=norm(A_fused)
    # A_fused= A_fused/norm_A;
    #CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true,true);
    init=initial_condition(init_type="PBC", reconstruct=true, has_AA_fused=false);

    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,[],ctm_setting,optim_setting)
    E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
    E=real(E_up+E_down)/3;
    println(E)
    global E_tem, CTM_tem, AA_fused_tem
    CTM_tem=deepcopy(CTM);
    AA_fused_tem=deepcopy(AA_fused);
    E_tem=deepcopy(E)
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

#fun(state_vec)

E,∂E,CTM_tem, AA_fused_tem=cfun(state_vec);
println(E,∂E)



dt=0.00001

E0=fun(state_vec);

grad=Vector{Float64}(undef,0);

for cc=1:length(state_vec)
    state_vec_tem=deepcopy(state_vec);
    state_vec_tem[cc]=state_vec_tem[cc]+dt;
    grad=vcat(grad,(fun(state_vec_tem)-E0)/dt);

end
println(grad)
println(∂E./grad)

