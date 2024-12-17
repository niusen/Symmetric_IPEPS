using Revise, TensorKit, Zygote
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)




include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\bosonic\\CTMRG_unitcell.jl")
include("..\\..\\src\\bosonic\\triangle\\triangle_spin_model_cell.jl")
include("..\\..\\src\\bosonic\\triangle\\triangle_spin_AD_cell.jl")
include("..\\..\\src\\bosonic\\triangle\\simple_update\\triangle_SimpleUpdate.jl")
include("..\\..\\src\\bosonic\\triangle\\simple_update\\triangle_SimpleUpdate_iPEPS.jl")

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################
# let
Random.seed!(1234);
import LinearAlgebra.BLAS as BLAS
# n_cpu=10;
# BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))

Dmax=9;


J=1;
K=1;
Φ=0.1;
parameters=Dict([("J", J),("K", K), ("Φ",  Φ)]);



trun_tol=1e-6;



chi=40;

"Unit-cell format:
ABABAB
CDCDCD
ABABAB
CDCDCD


A11  A21
A12  A22


actual unit-cell:
ABAB
BABA
ABAB
BABA
"

algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings

optim_setting=Optim_settings();
optim_setting.init_statenm="nothing";#"Gutzwiller_stochastic_iPESS_LS_D_6_chi_40_2.3592.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);


LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=10;
LS_ctm_setting.CTM_trun_tol=1e-8;
LS_ctm_setting.svd_lanczos_tol=1e-8;
LS_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
LS_ctm_setting.conv_check="singular_value";
LS_ctm_setting.CTM_ite_info=true;
LS_ctm_setting.CTM_conv_info=true;
LS_ctm_setting.CTM_trun_svd=false;
LS_ctm_setting.construct_double_layer=true;
LS_ctm_setting.grad_checkpoint=true;
dump(LS_ctm_setting);

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "triangle_SU4_spin";
dump(energy_setting);



##################################
"""
       /| s3
      / |
     /  |
    /   |
 s2 ----- s1
"""


##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;



##############


global Lx,Ly,A_cell
global chi, parameters, energy_setting, grad_ctm_setting


if optim_setting.init_statenm=="nothing"
    Vv=Rep[SU₂ × SU₂]((0,0)=>1, (0,1/2)=>1, (1/2,0)=>1, (1/2,1/2)=>1);
    Vp=Rep[SU₂ × SU₂]((1/2,1/2)=>1);
    T_set,lambdax_set,lambday_set=initial_iPEPS(Lx,Ly,Vp,Vv); 
    # B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS_uniform(Lx,Ly,V,Vp);    
else
    data=load(optim_setting.init_statenm);

    
    T_set=data["T_set"];
    lambdax_set=data["lambdax_set"];
    lambday_set=data["lambday_set"];
    
end








# A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
# init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
# CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,LS_ctm_setting);
# E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
# println(E_total)
# println(ex_set)
# println(ey_set)
# println(e_diagonala_set)
# println(e0_set)
# println(eU_set)

# D0set=[];
# for cc in eachindex(B_set)
#     D0set=vcat(D0set,[dim(space(B_set[cc],1)), dim(space(B_set[cc],2)), dim(space(B_set[cc],3))]);
# end
# D_max0=maximum(D0set);
# B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS_no_Hamiltonian(parameters, B_set, T_set, λ_set1, λ_set2, λ_set3, D_max0, trun_tol);

tau=1;
dt=0.1;


tol=dt*1e-3;#for determining convergence 
println("tau, dt="*string([tau,dt]))
Lx,Ly=size(T_set);
J=parameters["J"];
K=parameters["K"];
Φ=parameters["Φ"];
space_type=typeof(space(T_set[1,1],1));
###############
J1=J/2;
J_ijk=3*K*exp(im*Φ);
J_kji=3*K*exp(-im*Φ);
gate=gate_triangle(energy_setting,J1,J_ijk,J_kji,dt, space_type);


@time begin

px=1;py=1;
Lx,Ly=size(T_set);
# x_range=[mod1(plaquatte[1],Lx),mod1(plaquatte[1]+1,Ly)];
# y_range=[mod1(plaquatte[2],Lx),mod1(plaquatte[2]+1,Ly)];

# T_plaquatte=psi[x_range,y_range];


    pos1=[mod1(px+1,Lx),mod1(py-1,Ly)];
    pos2=[mod1(px+1,Lx),py];
    pos3=[px,py];
    T1=T_set[pos1[1],pos1[2]];
    T2=T_set[pos2[1],pos2[2]];
    T3=T_set[pos3[1],pos3[2]];
    λ1=lambdax_set[pos1[1],pos1[2]];
    λ2=lambdax_set[mod1(pos1[1]+1,Lx),pos1[2]];
    λ3=lambday_set[pos1[1],mod1(pos1[2]-1,Ly)];
    λ4=lambday_set[pos2[1],pos2[2]];
    λ5=lambdax_set[mod1(pos2[1]+1,Lx),pos2[2]];
    λ6=lambdax_set[pos3[1],pos3[2]];
    λ7=lambday_set[pos3[1],pos3[2]];
    λ8=lambday_set[pos3[1],mod1(pos3[2]-1,Ly)];

    @tensor T1[:]:=T1[1,-2,3,4,-5]*λ1[-1,1]*λ2[3,-3]*λ3[-4,4];
    @tensor T2[:]:=T2[-1,2,3,-4,-5]*λ4[2,-2]*λ5[3,-3];
    @tensor T3[:]:=T3[1,2,-3,4,-5]*λ6[-1,1]*λ7[2,-2]*λ8[-4,4];

    u1,s1,v1=tsvd(permute(T1,(1,3,4,),(2,5,)));#L1,R1,U1,newbond1,  newbond1,D1,d1   
    T1_left=u1;#L1,R1,U1,newbond1  
    T1_keep=s1*v1;#newbond1,D1,d1  

    u3,s3,v3=tsvd(permute(T3,(3,5,),(1,2,4,)));#R3,d3,newbond3,  newbond3,L3,D3,U3 
    T3_keep=u3*s3;#R3,d3,newbond3 
    T3_left=v3;#newbond3,L3,D3,U3 

    op=gate;
    @tensor T2_new[:]:=T1_keep[-4,1,-5]*T2[2,-2,-3,1,-6]*T3_keep[2,-7,-1];#newbond1,D1,d1,   L2,D2,R2,U2,d2,   R3,d3,newbond3   
    @tensor T2_new[:]:=T2_new[-1,-2,-3,-4,1,2,3]*op[-5,-6,-7,1,2,3];#newbond3,D2,R2,newbond1,d1,d2,d3

    u1,s1,v1=tsvd(permute(T2_new,(4,5,),(1,2,3,6,7,)); trunc=truncdim(Dmax));#newbond1,d1,    newbond3,D2,R2,d2,d3
    T1_keep=u1*sqrt(s1);#newbond1,d1,D1,
    @tensor T1_new[:]:=T1_left[-1,-3,-4,1]*T1_keep[1,-5,-2];#L1,R1,U1,newbond1,   newbond1,d1,D1

    T2T3=s1*v1;#U2,newbond3,D2,R2,d2,d3
    u3,s3,v3=tsvd(permute(T2T3,(1,3,4,5,),(2,6,)); trunc=truncdim(Dmax));#U2,D2,R2,d2,    newbond3,d3
    T2_new=permute(u3*sqrt(s3),(5,2,3,1,4,));
    T3_keep=sqrt(s3)*v3;#R3,newbond3,d3
    @tensor T3_new[:]:=T3_keep[-3,1,-5]*T3_left[1,-1,-2,-4];#R3,newbond3,d3    newbond3,L3,D3,U3 

    s1_inv_sqrt=sqrt(my_pinv(s1));
    @tensor T2_new[:]:=T2_new[-1,-2,-3,1,-5]*s1_inv_sqrt[-4,1];

    λ1_inv=my_pinv(λ1);
    λ2_inv=my_pinv(λ2);
    λ3_inv=my_pinv(λ3);
    λ4_inv=my_pinv(λ4);
    λ5_inv=my_pinv(λ5);
    λ6_inv=my_pinv(λ6);
    λ7_inv=my_pinv(λ7);
    λ8_inv=my_pinv(λ8);
    @tensor T1_new[:]:=T1_new[1,-2,3,4,-5]*λ1_inv[-1,1]*λ2_inv[3,-3]*λ3_inv[-4,4];
    @tensor T2_new[:]:=T2_new[-1,2,3,-4,-5]*λ4_inv[2,-2]*λ5_inv[3,-3];
    @tensor T3_new[:]:=T3_new[1,2,-3,4,-5]*λ6_inv[-1,1]*λ7_inv[2,-2]*λ8_inv[-4,4];

    s1=s1/norm(s1);
    s3=s3/norm(s3);
    lambday_set[pos1[1],pos1[2]]=sqrt(s1);
    lambdax_set[pos2[1],pos2[2]]=permute(sqrt(s3),(2,),(1,));

    T1_new=T1_new/norm(T1_new);
    T2_new=T2_new/norm(T2_new);
    T3_new=T3_new/norm(T3_new);
    T_set[pos1[1],pos1[2]]=T1_new;
    T_set[pos2[1],pos2[2]]=T2_new;
    T_set[pos3[1],pos3[2]]=T3_new;

    1+1

end

# tau=20;
# dt=0.02;
# #B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS(parameters, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);
# B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS_Hofstadter(energy_setting, parameters, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);

# tau=20;
# dt=0.002;
# B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS(parameters, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);



# A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
# init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
# CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,LS_ctm_setting);
# E_total,  ex_set, ey_set, e_diagonal_set, triangle_right_bot_set, triangle_left_top_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
# println(E_total)
# println(ex_set)
# println(ey_set)
# println(e_diagonal_set)
# println(triangle_right_bot_set)
# println(triangle_left_top_set)


# filenm="SU_iPESS_SU2_D"*string(D_max)*".jld2";
# jldsave(filenm; B_set, T_set, λ_set1, λ_set2, λ_set3)


# end
