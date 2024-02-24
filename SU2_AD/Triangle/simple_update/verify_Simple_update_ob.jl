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
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\simple_update\\verify_fermionic_triangle_SimpleUpdate_lib.jl")

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


t=1;
ϕ=pi/2;
μ=0;
U=0;
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);

D_max=6;
symmetric_hosvd=false;


global D_max, SU_trun_tol
SU_trun_tol=1e-8;
println("D_max= "*string(D_max))

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
optim_setting.init_statenm="Gutzwiller.jld2";#"Optim_cell_LS_D_4_chi_40_2.140901.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);


LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=50;
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
energy_setting.model = "spinful_triangle_lattice";
dump(energy_setting);



##################################
"""
       /| s3
      / |
     /  |
    /   |
 s2 ----- s1
"""

(Ident1,Ident2,), (N_occu1,N_occu2), (n_double1,n_double2,), (Cdag1,Cdag2,), (C1,C2,)=Hamiltonians_spinful_U1_SU2();

# H_Heisenberg, H123chiral, H12, H31, H23 =Hamiltonians();
# H_triangle=(J1/4)*H31+(J1/4)*H12+(J2/2)*H23+Jchi*H123chiral;


##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;

data=load(optim_setting.init_statenm);
x=data["x"];
function init_x(x)
    if size(x)==(2,1)
        TA=deepcopy(x[1].T);
        TB=deepcopy(x[2].T);
        TC=deepcopy(x[1].T);
        TD=deepcopy(x[2].T);
    elseif size(x)==(2,2)
        TA=deepcopy(x[1][1].T);
        TB=deepcopy(x[2][1].T);
        TC=deepcopy(x[1][2].T);
        TD=deepcopy(x[2][2].T);
    end
    return TA,TB,TC,TD
end

TA,TB,TC,TD=init_x(x);

# TA=LUdRD_to_LDRUd(TA);
# TB=LUdRD_to_LDRUd(TB);
# TC=LUdRD_to_LDRUd(TC);
# TD=LUdRD_to_LDRUd(TD);


λ_A_L=unitary(space(TA,1)',space(TA,1)');
λ_A_D=unitary(space(TA,2)',space(TA,2)'); 
λ_A_R=unitary(space(TA,3)',space(TA,3)');
λ_A_U=unitary(space(TA,4)',space(TA,4)');
λ_D_L=unitary(space(TD,1)',space(TD,1)');
λ_D_D=unitary(space(TD,2)',space(TD,2)'); 
λ_D_R=unitary(space(TD,3)',space(TD,3)');
λ_D_U=unitary(space(TD,4)',space(TD,4)');



######################

# tau=5;
# dt=0.1;
# TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U=itebd(TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U, H_triangle, trun_tol, tau, dt, D_max);

# tau=2;
# dt=0.05;
# TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U=itebd(TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U, H_triangle, trun_tol, tau, dt, D_max);

# tau=0.2;
# dt=0.01;
# TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U=itebd(TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U, H_triangle, trun_tol, tau, dt, D_max);


println(space(TA))
println(space(TB))
println(space(TC))
println(space(TD))

##############
@tensor TA[:]:=TA[1,2,3,4,-5]*λ_A_L[1,-1]*λ_A_D[2,-2]*λ_A_R[3,-3]*λ_A_U[4,-4];
@tensor TD[:]:=TD[1,2,3,4,-5]*λ_D_L[1,-1]*λ_D_D[2,-2]*λ_D_R[3,-3]*λ_D_U[4,-4];

# TA=LDRUd_to_LUdRD(TA);
# TB=LDRUd_to_LUdRD(TB);
# TC=LDRUd_to_LUdRD(TC);
# TD=LDRUd_to_LUdRD(TD);
##############
state_vec=Matrix{Square_iPEPS}(undef,2,2);
state_vec[1,1]=Square_iPEPS(TA);
state_vec[1,2]=Square_iPEPS(TC);
state_vec[2,1]=Square_iPEPS(TB);
state_vec[2,2]=Square_iPEPS(TD);

##############


global Lx,Ly,A_cell
A_cell=initial_tuple_cell(Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        global U_phy,A_cell
        A=state_vec[cx, cy].T;
        A_cell=fill_tuple(A_cell, A, cx,cy);
    end
end

global chi, parameters, energy_setting, grad_ctm_setting
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);


E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonians_spinful_U1_SU2();


dt=-0.1;
D_max=20;
hopping_coe_set=zeros(Lx,Ly)*im;
hopping_coe_set[1,1]=-1;
hopping_coe_set[2,1]=1;
hopping_coe_set[1,2]=-1;
hopping_coe_set[2,2]=1;
ob_set=zeros(Lx,Ly)*im;
Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=Hamiltonians_spinful_U1_SU2();
for cx=1:Lx;
    for cy=1:Ly;
        O1_set=(Ident_set[mod1(cx+1,Lx)],Cdag_set[mod1(cx+1,Lx)], C_set[mod1(cx+1,Lx)]);
        O2_set=(Ident_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)], Cdag_set[mod1(cx+2,Lx)]);
        ob=verify_evo_hopping_diagonala(CTM_cell,O1_set,O2_set,A_cell,AA_cell,cx,cy,hopping_coe_set[cx,cy],dt);
        ob_set[cx,cy]=ob;
    end
end
println(1.0.+(dt*hopping_coe_set.*e_diagonala_set+dt*conj(hopping_coe_set.*e_diagonala_set)))
println(ob_set)


#RU_LD_RD, t2 term
dt=0.1;
D_max=20;
tx_coe_set=[im,im];
ty_coe_set=[-1,1];
t2_coe_set=[-1,1];
U_coe=1;
ob_set=zeros(Lx,Ly)*im;
include("..\\..\\src\\fermionic\\simple_update\\verify_fermionic_triangle_SimpleUpdate_lib.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
for cx=1:Lx;
    for cy=1:Ly;

        ####################
        O1=Cdag_set[mod1(cx+1,Lx)];
        O2=C_set[mod1(cx+2,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*tx_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh_tx[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        ######################
        O1=Cdag_set[mod1(cx+2,Lx)];
        O2=C_set[mod1(cx+2,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*ty_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(Cdag_set[mod1(cx+1,Lx)],2),space(Cdag_set[mod1(cx+1,Lx)],2));
        @tensor hh_ty[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        #####################
        O1=Cdag_set[mod1(cx+1,Lx)];
        O2=C_set[mod1(cx+2,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=-op;#!!!!!!! somehow this minus sign is required
        op=op*t2_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh[:]:=hh[-1,-3,-4,-6]*Id[-2,-5];
        sgate=swap_gate(hh,2,3);
        @tensor hh_t2[:]:=sgate[-2,-3,1,2]*hh[-1,1,2,-4,3,4]*sgate'[3,4,-5,-6];
        #################
        OU_LD=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        OU_RU=n_double_set[mod1(cx+2,Lx)]-(1/2)*N_occu_set[mod1(cx+2,Lx)]+(1/4)*Ident_set[mod1(cx+2,Lx)];
        OU_RD=n_double_set[mod1(cx+2,Lx)]-(1/2)*N_occu_set[mod1(cx+2,Lx)]+(1/4)*Ident_set[mod1(cx+2,Lx)];
        Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
        Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
        Id_RD=unitary(space(OU_RD,1),space(OU_RD,1));
        @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
        @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
        @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
        hh_U=(hh_LD+hh_RU+hh_RD)*U_coe;
        
        #################
        hh=permute(hh_tx+hh_ty+hh_t2+hh_U,(1,2,3,),(4,5,6,));#hh_tx+hh_ty+hh_t2+hh_U
        eu,ev=eigh(hh);
        gate=ev*exp(-dt*eu)*ev';
        ob=verify_evo_hopping_RU_LD_RD(CTM_cell,gate,A_cell,AA_cell,cx,cy);#op_LD_RD_RU
        ob_set[cx,cy]=ob;
    end
end
m_ex=(-dt*tx_coe_set.*ex_set-dt*conj(tx_coe_set.*ex_set))
m_ey=(-dt*ty_coe_set.*ey_set-dt*conj(ty_coe_set.*ey_set))
m_e_t2=(-dt*t2_coe_set.*e_diagonala_set-dt*conj(t2_coe_set.*e_diagonala_set))
m_U=exp.(-dt*eU_set)
println(exp.(m_ex+m_ey+m_e_t2+m_U))
println(ob_set)



#LU_RU_LD, t2 term
dt=0.1;
D_max=20;
tx_coe_set=[im,im];
ty_coe_set=[1,-1];
t2_coe_set=[-1,1];
U_coe=1;
ob_set=zeros(Lx,Ly)*im;
include("..\\..\\src\\fermionic\\simple_update\\verify_fermionic_triangle_SimpleUpdate_lib.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
for cx=1:Lx;
    for cy=1:Ly;
        
        ####################
        O1=Cdag_set[mod1(cx+1,Lx)];
        O2=C_set[mod1(cx+2,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*tx_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,1),space(hh,1));
        @tensor hh_tx[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        ######################
        O1=Cdag_set[mod1(cx+1,Lx)];
        O2=C_set[mod1(cx+1,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*ty_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(Cdag_set[mod1(cx+2,Lx)],2),space(Cdag_set[mod1(cx+2,Lx)],2));
        @tensor hh_ty[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        #####################
        O1=Cdag_set[mod1(cx+1,Lx)];
        O2=C_set[mod1(cx+2,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=-op;#!!!!!!! somehow this minus sign is required
        op=op*t2_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,1),space(hh,1));
        @tensor hh[:]:=hh[-1,-3,-4,-6]*Id[-2,-5];
        sgate=swap_gate(hh,2,3);
        @tensor hh_t2[:]:=sgate[-2,-3,1,2]*hh[-1,1,2,-4,3,4]*sgate'[3,4,-5,-6];##op_LD_LU_RU
        #################
        OU_LD=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        OU_RU=n_double_set[mod1(cx+2,Lx)]-(1/2)*N_occu_set[mod1(cx+2,Lx)]+(1/4)*Ident_set[mod1(cx+2,Lx)];
        OU_LU=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
        Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
        Id_LU=unitary(space(OU_LU,1),space(OU_LU,1));
        @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_LU[-2,-5]*Id_RU[-3,-6];
        @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_LU[-2,-5]*OU_RU[-3,-6];
        @tensor hh_LU[:]:=Id_LD[-1,-4]*OU_LU[-2,-5]*Id_RU[-3,-6];
        # hh_U=(hh_LD+hh_RU+hh_LU)*U_coe;
        hh_U=(hh_LU)*U_coe;
                

        ###############################
        hh=permute(hh_tx+hh_ty+hh_t2+hh_U,(1,2,3,),(4,5,6,));#hh_tx+hh_ty+hh_t2+hh_U
        eu,ev=eigh(hh);
        gate=ev*exp(-dt*eu)*ev';
        ob=verify_evo_hopping_LU_RU_LD(CTM_cell,gate,A_cell,AA_cell,cx,cy);#op_LD_LU_RU
        ob_set[cx,cy]=ob;
    end
end
m_ex=(-dt*tx_coe_set.*ex_set-dt*conj(tx_coe_set.*ex_set))
m_ey=(-dt*ty_coe_set.*ey_set[2:-1:1,:]-dt*conj(ty_coe_set.*ey_set[2:-1:1,:]));
m_e_t2=(-dt*t2_coe_set.*e_diagonala_set-dt*conj(t2_coe_set.*e_diagonala_set))
m_U=(-dt*eU_set)
println(exp.(m_ex+m_ey+m_e_t2+m_U))
println(ob_set)



dt=-1;
D_max=30;
hopping_coe_set=zeros(Lx,Ly)*im;
hopping_coe_set[1,1]=im;
hopping_coe_set[2,1]=im;
hopping_coe_set[1,2]=im;
hopping_coe_set[2,2]=im;
ob_set=zeros(Lx,Ly)*im;
Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=Hamiltonians_spinful_U1_SU2();
for cx=1:Lx;
    for cy=1:Ly;
        O1_set=(Ident_set[mod1(cx+1,Lx)],Cdag_set[mod1(cx+1,Lx)], C_set[mod1(cx+1,Lx)]);
        O2_set=(Ident_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)], Cdag_set[mod1(cx+2,Lx)]);
        ob=verify_evo_hopping_x(CTM_cell,O1_set,O2_set,A_cell,AA_cell,cx,cy,hopping_coe_set[cx,cy],dt);
        ob_set[cx,cy]=ob;
    end
end
println(1.0.+(dt*hopping_coe_set.*ex_set+dt*conj(hopping_coe_set.*ex_set)))
println(ob_set)


dt=-1;
D_max=30;
hopping_coe_set=zeros(Lx,Ly)*im;
hopping_coe_set[1,1]=im;
hopping_coe_set[2,1]=im;
hopping_coe_set[1,2]=im;
hopping_coe_set[2,2]=im;
ob_set=zeros(Lx,Ly)*im;
include("..\\..\\src\\fermionic\\simple_update\\verify_fermionic_triangle_SimpleUpdate_lib.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
for cx=1:Lx;
    for cy=1:Ly;
        O1_set=(Ident_set[mod1(cx+1,Lx)],Cdag_set[mod1(cx+1,Lx)],);
        O2_set=(Ident_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)],);
        ob=verify_evo_hopping_LU_RU(CTM_cell,O1_set,O2_set,A_cell,AA_cell,cx,cy,hopping_coe_set[cx,cy],dt);
        ob_set[cx,cy]=ob;
    end
end
println(1.0.+(dt*hopping_coe_set.*ex_set+dt*conj(hopping_coe_set.*ex_set)))
println(ob_set)




dt=-1;
D_max=20;
hopping_coe_set=zeros(Lx,Ly)*im;
hopping_coe_set[1,1]=-1;
hopping_coe_set[2,1]=1;
hopping_coe_set[1,2]=-1;
hopping_coe_set[2,2]=1;
ob_set=zeros(Lx,Ly)*im;
Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=Hamiltonians_spinful_U1_SU2();
for cx=1:Lx;
    for cy=1:Ly;
        O1_set=(Ident_set[mod1(cx+2,Lx)],Cdag_set[mod1(cx+2,Lx)], C_set[mod1(cx+2,Lx)]);
        O2_set=(Ident_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)], Cdag_set[mod1(cx+2,Lx)]);
        ob=verify_evo_hopping_y(CTM_cell,O1_set,O2_set,A_cell,AA_cell,cx,cy,hopping_coe_set[cx,cy],dt);
        ob_set[cx,cy]=ob;
    end
end
println(1.0.+(dt*hopping_coe_set.*ey_set+dt*conj(hopping_coe_set.*ey_set)))
println(ob_set)




dt=-1;
D_max=6;
h_coe_set=zeros(Lx,Ly)*im;
U=1;
h_coe_set[1,1]=U;
h_coe_set[2,1]=U;
h_coe_set[1,2]=U;
h_coe_set[2,2]=U;
ob_set=zeros(Lx,Ly)*im;
Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=Hamiltonians_spinful_U1_SU2();
for cx=1:Lx;
    for cy=1:Ly;
        O1_set=(Ident_set[mod1(cx+1,Lx)],n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)]);
        ob=verify_evo_onsite(CTM_cell,O1_set,A_cell,AA_cell,cx,cy,h_coe_set[cx,cy],dt);
        ob_set[cx,cy]=ob;
    end
end
println(1.0.+(dt*h_coe_set.*eU_set))
println(ob_set)

# filenm="SU_D_"*string(D_max)*".jld2"
# jldsave(filenm;x=state_vec);

# mat_filenm="SU_D_"*string(D_max)*".mat"
# matwrite(mat_filenm, Dict(
#     "E_plaquatte_cell" => E_plaquatte_cell,
#     "space_Tu" => string(codomain(T_u)),
#     "space_Td" => string(codomain(T_d))
# ); compress = false)


