using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit

using LinearAlgebra, OptimKit
#using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using LineSearches
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)

include("..\\src\\square_spin_operator.jl")
include("..\\src\\iPEPS_ansatz.jl")
include("..\\src\\CTMRG.jl")
include("..\\src\\CTMRG_unitcell.jl")
include("..\\src\\square_model.jl")
include("..\\src\\square_model_cell.jl")

include("..\\src\\Settings.jl")
include("..\\src\\Settings_cell.jl")
include("..\\src\\AD_lib.jl")
include("..\\src\\line_search_lib.jl")
include("..\\src\\line_search_lib_cell.jl")
include("..\\src\\optimkit_lib.jl")

include("..\\src\\square_SimpleUpdate_lib.jl")
include("..\\src\\square_RVB_ansatz.jl")

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################

Random.seed!(1234)
symmetric_initial=false;
J1=1.78;
J2=0.84;
Jchi=0.375*2*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
D_max=3;
symmetric_hosvd=false;
trun_tol=1e-6;


println("D_max= "*string(D_max))

chi=80;

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

LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=100;
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

energy_setting=Square_Energy_settings();
energy_setting.model = "triangle_J1_J2_Jchi";
dump(energy_setting);





##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;

A=RVB_ansatz(1,1,im);
state=Square_iPEPS(A);




##############
state_vec=Matrix{Square_iPEPS}(undef,2,2);
state_vec[1,1]=Square_iPEPS(A);
state_vec[1,2]=Square_iPEPS(A);
state_vec[2,1]=Square_iPEPS(A);
state_vec[2,2]=Square_iPEPS(A);

##############


global Lx,Ly,A_cell
A_cell=initial_tuple_cell(Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        global A_cell
        A=state_vec[cx, cy].T;
        A_cell=fill_tuple(A_cell, A, cx,cy);
    end
end

global chi, parameters, energy_setting, grad_ctm_setting



init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell,chi,init,[],LS_ctm_setting);
E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(real.(E_LU_RU_LD_set))
println(real.(E_LD_RU_RD_set))
println(real.(E_LU_LD_RD_set))
println(real.(E_LU_RU_RD_set))

corner="LU";#"LU","RD"

Vp=SU2Space(0=>1,1/2=>1,1=>1);
U=TensorMap(randn,Vp,space(A,1));
# U,S,V=tsvd(U);

for ind=1:4
    if corner=="LU"
        if ind==1
            pos1=[1,1];
            pos2=[2,1];

            A_cell_tem=deepcopy(A_cell);
            @tensor A_tem1[:]:=A[1,-2,-3,-4,-5]*U[-1,1];
            @tensor A_tem2[:]:=A[-1,-2,1,-4,-5]*U'[1,-3];
            A_cell_tem=fill_tuple(A_cell_tem, A_tem1, pos1[1],pos1[2]);
            A_cell_tem=fill_tuple(A_cell_tem, A_tem2, pos2[1],pos2[2]);

            CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_tem,chi,init,[],LS_ctm_setting);
            E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell_tem::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
            println(real.(E_LU_RU_LD_set))
            println(real.(E_LD_RU_RD_set))
            println(real.(E_LU_LD_RD_set))
            println(real.(E_LU_RU_RD_set))
        elseif ind==2
            pos1=[1,1];
            pos2=[1,2];

            A_cell_tem=deepcopy(A_cell);
            @tensor A_tem1[:]:=A[-1,1,-3,-4,-5]*U[-2,1];
            @tensor A_tem2[:]:=A[-1,-2,-3,1,-5]*U'[1,-4];
            A_cell_tem=fill_tuple(A_cell_tem, A_tem1, pos1[1],pos1[2]);
            A_cell_tem=fill_tuple(A_cell_tem, A_tem2, pos2[1],pos2[2]);

            CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_tem,chi,init,[],LS_ctm_setting);
            E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell_tem::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
            println(real.(E_LU_RU_LD_set))
            println(real.(E_LD_RU_RD_set))
            println(real.(E_LU_LD_RD_set))
            println(real.(E_LU_RU_RD_set))
        elseif ind==3
            pos1=[1,1];
            pos2=[2,1];

            A_cell_tem=deepcopy(A_cell);
            @tensor A_tem1[:]:=A[-1,-2,1,-4,-5]*U'[1,-3];
            @tensor A_tem2[:]:=A[1,-2,-3,-4,-5]*U[-1,1];
            A_cell_tem=fill_tuple(A_cell_tem, A_tem1, pos1[1],pos1[2]);
            A_cell_tem=fill_tuple(A_cell_tem, A_tem2, pos2[1],pos2[2]);

            CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_tem,chi,init,[],LS_ctm_setting);
            E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell_tem::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
            println(real.(E_LU_RU_LD_set))
            println(real.(E_LD_RU_RD_set))
            println(real.(E_LU_LD_RD_set))
            println(real.(E_LU_RU_RD_set))
        elseif ind==4
            pos1=[1,1];
            pos2=[1,2];

            A_cell_tem=deepcopy(A_cell);
            @tensor A_tem1[:]:=A[-1,-2,-3,1,-5]*U'[1,-4];
            @tensor A_tem2[:]:=A[-1,1,-3,-4,-5]*U[-2,1];
            A_cell_tem=fill_tuple(A_cell_tem, A_tem1, pos1[1],pos1[2]);
            A_cell_tem=fill_tuple(A_cell_tem, A_tem2, pos2[1],pos2[2]);

            CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_tem,chi,init,[],LS_ctm_setting);
            E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell_tem::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
            println(real.(E_LU_RU_LD_set))
            println(real.(E_LD_RU_RD_set))
            println(real.(E_LU_LD_RD_set))
            println(real.(E_LU_RU_RD_set))
        end
    elseif corner=="RD"
        if ind==1
            pos1=[2,2];
            pos2=[1,2];

            A_cell_tem=deepcopy(A_cell);
            @tensor A_tem1[:]:=A[1,-2,-3,-4,-5]*U[-1,1];
            @tensor A_tem2[:]:=A[-1,-2,1,-4,-5]*U'[1,-3];
            A_cell_tem=fill_tuple(A_cell_tem, A_tem1, pos1[1],pos1[2]);
            A_cell_tem=fill_tuple(A_cell_tem, A_tem2, pos2[1],pos2[2]);

            CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_tem,chi,init,[],LS_ctm_setting);
            E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell_tem::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
            println(real.(E_LU_RU_LD_set))
            println(real.(E_LD_RU_RD_set))
            println(real.(E_LU_LD_RD_set))
            println(real.(E_LU_RU_RD_set))
        elseif ind==2
            pos1=[2,2];
            pos2=[2,1];

            A_cell_tem=deepcopy(A_cell);
            @tensor A_tem1[:]:=A[-1,1,-3,-4,-5]*U[-2,1];
            @tensor A_tem2[:]:=A[-1,-2,-3,1,-5]*U'[1,-4];
            A_cell_tem=fill_tuple(A_cell_tem, A_tem1, pos1[1],pos1[2]);
            A_cell_tem=fill_tuple(A_cell_tem, A_tem2, pos2[1],pos2[2]);

            CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_tem,chi,init,[],LS_ctm_setting);
            E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell_tem::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
            println(real.(E_LU_RU_LD_set))
            println(real.(E_LD_RU_RD_set))
            println(real.(E_LU_LD_RD_set))
            println(real.(E_LU_RU_RD_set))
        elseif ind==3
            pos1=[2,2];
            pos2=[1,2];

            A_cell_tem=deepcopy(A_cell);
            @tensor A_tem1[:]:=A[-1,-2,1,-4,-5]*U'[1,-3];
            @tensor A_tem2[:]:=A[1,-2,-3,-4,-5]*U[-1,1];
            A_cell_tem=fill_tuple(A_cell_tem, A_tem1, pos1[1],pos1[2]);
            A_cell_tem=fill_tuple(A_cell_tem, A_tem2, pos2[1],pos2[2]);

            CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_tem,chi,init,[],LS_ctm_setting);
            E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell_tem::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
            println(real.(E_LU_RU_LD_set))
            println(real.(E_LD_RU_RD_set))
            println(real.(E_LU_LD_RD_set))
            println(real.(E_LU_RU_RD_set))
        elseif ind==4
            pos1=[2,2];
            pos2=[2,1];

            A_cell_tem=deepcopy(A_cell);
            @tensor A_tem1[:]:=A[-1,-2,-3,1,-5]*U'[1,-4];
            @tensor A_tem2[:]:=A[-1,1,-3,-4,-5]*U[-2,1];
            A_cell_tem=fill_tuple(A_cell_tem, A_tem1, pos1[1],pos1[2]);
            A_cell_tem=fill_tuple(A_cell_tem, A_tem2, pos2[1],pos2[2]);

            CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_tem,chi,init,[],LS_ctm_setting);
            E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell_tem::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
            println(real.(E_LU_RU_LD_set))
            println(real.(E_LD_RU_RD_set))
            println(real.(E_LU_LD_RD_set))
            println(real.(E_LU_RU_RD_set))
        end
    end
end






