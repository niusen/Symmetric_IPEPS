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
include("resource_codes\\kagome_CTMRG_test.jl")
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
ctm_setting.CTM_ite_nums=1;
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



a=load("CTM_ite.jld2");

Cset=a["Cset"];
Tset=a["Tset"];
AA=a["AA"];
chi=a["chi"];
direction=a["direction"];


function ob_1site_closed_test(C1,C2,C3,C4,T1,T2,T3,T4,  AA_fused)
    
    @tensor envL[:]:=C1[1,-1]*T4[2,-2,1]*C4[-3,2];
    @tensor envR[:]:=C2[-1,1]*T2[1,-2,2]*C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*T1[1,3,-1]*AA_fused[2,5,-2,3]*T3[-3,5,4];
    Norm=@tensor envL[1,2,3]*envR[1,2,3];
    
    return Norm;
end


function fun(x)
    global chi, ipess_irrep, elementary_tensors,parameters
    
    #Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=vector_to_coes(elementary_tensors, ipess_irrep, x);

    bond_tensor,triangle_tensor=construct_su2_PG_IPESS_vec(x,elementary_tensors, ipess_irrep);
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;

    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

    AA, U_L,U_D,U_R,U_U=build_double_layer(A_fused,[]);




    #global Cset,Tset,chi,direction
    CTM=init_CTM(chi,A_fused,"PBC",false); Cset=CTM.Cset;Tset=CTM.Tset;


    M1=get_Cset(Cset, mod1(direction,4));
    M2=get_Tset(Tset, mod1(direction,4));
    M3=get_Tset(Tset, mod1(direction-1,4));
    #M4=AA;
    M5=M3;
    #M6=M4;
    M7=get_Cset(Cset, mod1(direction-1,4));
    M8=get_Tset(Tset, mod1(direction-2,4));


    @tensor MMup[:]:=M1[1,2]*M2[2,3,-3]*M3[-1,4,1]*AA[4,-2,-4,3];
    @tensor MMlow[:]:=M5[1,3,-1]*AA[3,4,-4,-2]*M7[2,1]*M8[-3,4,2];


    #define permute index that is heavily used
    permut_ind1=[];
    permut_ind2=[];

    permut_ind1=(1,2,);
    permut_ind2=(3,4,);


    MMup=permute(MMup,permut_ind1,permut_ind2)
    MMlow=permute(MMlow,permut_ind1,permut_ind2)


    M1_=get_Cset(Cset, mod1(direction+1,4));
    M2_=get_Tset(Tset, mod1(direction,4));
    M3_=get_Tset(Tset, mod1(direction+1,4));
    #M4_=M4;
    M5_=M3_;
    #M6_=M4_;
    M7_=get_Cset(Cset, mod1(direction-2,4));
    M8_=get_Tset(Tset, mod1(direction-2,4));


    @tensor MMup_reflect[:]:=M2_[-1,3,1]* M1_[1,2]* AA[-2,-4,4,3]* M3_[2,4,-3];
    @tensor MMlow_reflect[:]:=M5_[-4,-3,2]*M8_[1,-2,-1]*M7_[2,1];
    @tensor MMlow_reflect[:]:=MMlow_reflect[-1,1,2,-3]*AA[-2,1,2,-4];

    MMup_reflect=permute(MMup_reflect,permut_ind1,permut_ind2)
    MMlow_reflect=permute(MMlow_reflect,permut_ind1,permut_ind2)


    

    RMup=permute(MMup*MMup_reflect,permut_ind2,permut_ind1);
    RMlow=MMlow*MMlow_reflect;


    #without the below normalization, the gradiant of svd will explode!!!
    #Also we should ignore derivative of this step, otherwise it seems that the normalization factor will accumulate and the grad explode again!!!
    
    # RMlow_norm=norm(RMlow);
    # RMlow= RMlow/RMlow_norm;

    # RMup_norm=@ignore_derivatives norm(RMup);
    # RMup= RMup/RMup_norm;

    RMlow=@ignore_derivatives RMlow/norm(RMlow);
    RMup=@ignore_derivatives RMup/norm(RMup);

    M=RMup*RMlow;
    #M=@ignore_derivatives M/norm(M);

    
    #jldsave("hard_tensor.jld2"; M)

 
    #uM,sM,vM = tsvd(M; trunc=truncdim(chi+20));
    
    # @ignore_derivatives Random.seed!(12345)
    # M0=@ignore_derivatives TensorMap(randn,space(M,1)*space(M,2),space(M,1)*space(M,2));M0=@ignore_derivatives M0/norm(M0)
    uM,sM,vM = tsvd(M);

    
    # println(sM.data.values)
    sM_norm=norm(sM);
    sM=sM/sM_norm;
    # println(sM.data.values)treat_svd_results
    multiplet_tol=1e-5;
    trun_tol=1e-8;
    #uM,sM,vM,sM_inv_sqrt=treat_svd_results(uM,sM,vM,chi,multiplet_tol,trun_tol);
    #sM_inv_sqrt=sdiag_inv_sqrt(sM);
    sM_inv_sqrt=@ignore_derivatives unitary(space(sM,1),space(sM,1));# No inverse operation actually


    PM_inv=RMlow*vM'*sM_inv_sqrt;
    PM=sM_inv_sqrt*uM'*RMup;


    # PM_inv=@ignore_derivatives unitary(space(RMlow,1)*space(RMlow,2), fuse(space(RMlow,1)*space(RMlow,2)))
    # PM=@ignore_derivatives unitary(fuse(space(RMup,3)*space(RMup,4)), space(RMup,3)'*space(RMup,4)')

    # println(space(PM_inv_))
    # println(space(PM_inv))

    # println(space(PM_))
    # println(space(PM))

    PM=permute(PM,(permut_ind1.+1),(1,));
    #println([norm(PM),norm(PM_inv)])




    @tensor M5tem[:]:=get_Tset(Tset, mod1(direction-1,4))[4,3,1]*AA[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
    @tensor M1tem[:]:=get_Cset(Cset, mod1(direction,4))[1,2]*get_Tset(Tset, mod1(direction,4))[2,3,-2]*PM_inv[1,3,-1];
    @tensor M7tem[:]:=get_Cset(Cset, mod1(direction-1,4))[1,2]*get_Tset(Tset, mod1(direction-2,4))[-1,3,1]* PM[2,3,-2];



    norm_M1=norm(M1tem);
    C1_tem= M1tem/norm_M1; #somehow I must ignore grad of such normalization, otherwise error will occur in the checkpoint punction

    T_norm=norm(M5tem);
    T4_tem= M5tem/T_norm;
    
    norm_M7=norm(M7tem);
    C4_tem= M7tem/norm_M7;
    

    # Cset=set_Cset(Cset, C1_tem,mod1(direction,4));
    # Cset=set_Cset(Cset, C4_tem, mod1(direction-1,4));
    # Tset=set_Tset(Tset, T4_tem, mod1(direction-1,4));

    H_triangle, H_Heisenberg, H12_tensorkit, H31_tensorkit, H23_tensorkit=@ignore_derivatives Hamiltonians(U_phy,parameters["J1"],parameters["J2"],parameters["J3"],parameters["Jchi"],parameters["Jtrip"])
    AA_H,_=build_double_layer(A_fused,H_triangle);

    if direction==1
        Norm=ob_1site_closed_test(C1_tem, Cset.C2, Cset.C3, C4_tem, Tset.T1, Tset.T2, Tset.T3, T4_tem,   AA);
        #E=ob_1site_closed_test(C1_tem, Cset.C2, Cset.C3, C4_tem, Tset.T1, Tset.T2, Tset.T3, T4_tem,   AA_H);
        E=real(Norm)
        #E=real(E/Norm)
        #println(E*2/3)
        return E
    elseif direction==2
        return C4_tem, C1_tem, Cset.C3, Cset.C4, T4_tem, Tset.T2, Tset.T3, Tset.T4
    elseif direction==3
        return Cset.C1, C4_tem, C1_tem, Cset.C4, Tset.T1, T4_tem, Tset.T3, Tset.T4
    elseif direction==4
        return Cset.C1, Cset.C2, C4_tem, C1_tem, Tset.T1, Tset.T2, T4_tem, Tset.T4
    end

end


function cfun(state_vec)
    
    ∂E = fun'(state_vec)
    #E=fun(state_vec)
    
    @assert !isnan(norm(∂E))
    
    return ∂E
end

A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
global A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb  
json_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd=initial_state(Bond_irrep,Triangle_irrep,nonchiral,D,optim_setting.init_statenm,optim_setting.init_noise)
elementary_tensors=Elementary_tensors(A_set,B_set,A1_set,A2_set,A1_has_odd,A2_has_odd);
bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb)



state_vec=coes_to_vector(Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, ipess_irrep)

state_vec=normalize_IPESS_SU2_PG_vec(elementary_tensors, ipess_irrep, state_vec);




# fun(state_vec)


∂E=cfun(state_vec);
println(∂E)



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


