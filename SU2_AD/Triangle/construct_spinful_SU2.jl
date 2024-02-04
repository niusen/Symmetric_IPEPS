using Revise, TensorKit
using LinearAlgebra, OptimKit
using TensorKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
using Random
using LineSearches
using Dates
cd(@__DIR__)

include("..\\src\\bosonic\\Settings.jl")
include("..\\src\\bosonic\\Settings_cell.jl")
include("..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\src\\bosonic\\AD_lib.jl")
include("..\\src\\bosonic\\line_search_lib.jl")
include("..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\src\\bosonic\\optimkit_lib.jl")
include("..\\src\\bosonic\\CTMRG.jl")
include("..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\src\\fermionic\\swap_funs.jl")
include("..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\src\\fermionic\\double_layer_funs.jl")
include("..\\src\\fermionic\\square_Hubbard_AD_cell.jl")

include("..\\src\\fermionic\\fermi_permute.jl")

Random.seed!(777)

D=2;
chi=40

t=1;
ϕ=pi/2;
μ=0;
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ)]);



grad_ctm_setting=grad_CTMRG_settings();
grad_ctm_setting.CTM_conv_tol=1e-6;
grad_ctm_setting.CTM_ite_nums=0;
grad_ctm_setting.CTM_trun_tol=1e-8;
grad_ctm_setting.svd_lanczos_tol=1e-8;
grad_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
grad_ctm_setting.conv_check="singular_value";
grad_ctm_setting.CTM_ite_info=true;
grad_ctm_setting.CTM_conv_info=true;
grad_ctm_setting.CTM_trun_svd=false;
grad_ctm_setting.construct_double_layer=true;
grad_ctm_setting.grad_checkpoint=true;
dump(grad_ctm_setting);

LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=0;
LS_ctm_setting.CTM_trun_tol=1e-8;
LS_ctm_setting.svd_lanczos_tol=1e-8;
LS_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
LS_ctm_setting.conv_check="singular_value";
LS_ctm_setting.CTM_ite_info=false;
LS_ctm_setting.CTM_conv_info=true;
LS_ctm_setting.CTM_trun_svd=false;
LS_ctm_setting.construct_double_layer=true;
LS_ctm_setting.grad_checkpoint=true;
dump(LS_ctm_setting);

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);

optim_setting=Optim_settings();
optim_setting.init_statenm="Optim_cell_LS_D_2_chi_20_1.081059.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "spinless_triangle_lattice";
dump(energy_setting);

algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings


global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings


global Vv_set

if D==2
    Vv=Rep[ℤ₂](0=>1,1=>1); 
elseif D==4

elseif D==6
    
end







global Lx,Ly
Lx=2;
Ly=1;






init_complex_tensor=true;

state_vec=initial_fPEPS_state_spinless_Z2(Vv, optim_setting.init_statenm, optim_setting.init_noise,init_complex_tensor)
state_vec=normalize_tensor_group(state_vec);

A1=state_vec[1].T;
A2=state_vec[2].T;

A1_dense=convert(Array,A1);
A2_dense=convert(Array,A2);


function construct_spin(A)
    A=permute(A,(1,4,5,3,2,));#LUd|><|RD

    U=unitary(fuse(space(A,1)*space(A,1)), space(A,1)*space(A,1));
    @tensor A_spin[:]:=A[-1,-2,-3,-4,-5]*A[-6,-7,-8,-9,-10];#Lup,Uup,dup,Rup,Dup,  Ldn,Udn,ddn,Rdn,Ddn
    A_spin=permute_neighbour_ind(A_spin,5,6,10);#Lup,Uup,dup,Rup,Ldn,Dup,Udn,ddn,Rdn,Ddn
    A_spin=permute_neighbour_ind(A_spin,4,5,10);#Lup,Uup,dup,Ldn,Rup,Dup,Udn,ddn,Rdn,Ddn
    A_spin=permute_neighbour_ind(A_spin,3,4,10);#Lup,Uup,Ldn,dup,Rup,Dup,Udn,ddn,Rdn,Ddn
    A_spin=permute_neighbour_ind(A_spin,2,3,10);#Lup,Ldn,Uup,dup,Rup,Dup,Udn,ddn,Rdn,Ddn
    @tensor A_spin[:]:=A_spin[1,2,-2,-3,-4,-5,-6,-7,-8,-9]*U[-1,1,2]; #L,Uup,dup,Rup,Dup,Udn,ddn,Rdn,Ddn

    A_spin=permute_neighbour_ind(A_spin,5,6,9);#L,Uup,dup,Rup,Udn,Dup,ddn,Rdn,Ddn
    A_spin=permute_neighbour_ind(A_spin,4,5,9);#L,Uup,dup,Udn,Rup,Dup,ddn,Rdn,Ddn
    A_spin=permute_neighbour_ind(A_spin,3,4,9);#L,Uup,Udn,dup,Rup,Dup,ddn,Rdn,Ddn
    @tensor A_spin[:]:=A_spin[-1,1,2,-3,-4,-5,-6,-7,-8]*U[-2,1,2]; #L,U,dup,Rup,Dup,ddn,Rdn,Ddn

    A_spin=permute_neighbour_ind(A_spin,5,6,8);#L,U,dup,Rup,ddn,Dup,Rdn,Ddn
    A_spin=permute_neighbour_ind(A_spin,4,5,8);#L,U,dup,ddn,Rup,Dup,Rdn,Ddn
    @tensor A_spin[:]:=A_spin[-1,-2,1,2,-4,-5,-6,-7]*U[-3,1,2];#L,U,d,Rup,Dup,Rdn,Ddn

    A_spin=permute_neighbour_ind(A_spin,5,6,7);#L,U,d,Rup,Rdn,Dup,Ddn
    A_spin=permute_neighbour_ind(A_spin,4,5,7);#L,U,d,Rdn,Rup,Dup,Ddn    note that an unwanted permute is used
    @tensor A_spin[:]:=A_spin[-1,-2,-3,1,2,-5,-6]*U'[2,1,-4];#L,U,d,R,Dup,Ddn    be careeful about order

    A_spin=permute_neighbour_ind(A_spin,5,6,6);#L,U,d,R,Ddn,Dup    note that an unwanted permute is used
    @tensor A_spin[:]:=A_spin[-1,-2,-3,-4,1,2]*U'[2,1,-5]; #L,U,d,R,D  be careeful about order

    A_spin=permute(A_spin,(1,5,4,2,3,));
    return A_spin
end

A1_new=construct_spin(A1);
A2_new=construct_spin(A2);

Vv=SU2Space(0=>2,1/2=>1);
A1_new=TensorMap(convert(Array,A1_new),Vv*Vv'*Vv'*Vv,SU2Space(0=>2,1/2=>1)');

#remark: I don't know how to construct su2 tensor. The direct tensor product seems to be not su2 invariant.


A_cell=initial_tuple_cell(Lx,Ly);
A_cell=fill_tuple(A_cell, A1_new, 1,1);
A_cell=fill_tuple(A_cell, A2_new, 2,1);


init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init,[],grad_ctm_setting);


##########################
Ident, N_occu, n_double, Cdag, C =@ignore_derivatives Hamiltonians_spinful_Z2();
#########################
Vdummy=Rep[ℤ₂](1=>1);
V=Rep[ℤ₂](0=>2,1=>2);


Id=[1.0 0;0 1.0];
sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];

#order of kron() command: (0,0), (0,1), (1,0), (1,1)
order=[1,4,3,2];

Ident=kron(Id,Id);
Ident=TensorMap(Ident[order,order],  V  ←  V);

N_occu=kron(occu,Id)+kron(Id,occu);
N_occu=TensorMap(N_occu[order,order],  V ←  V);
n_double=kron(occu,occu)
n_double=TensorMap(n_double[order,order],  V ←  V);

# method 1
Cdagup=zeros(4,4,1);
Cdagup[order,order,1]=kron(sp,Id);
Cdagdn=zeros(4,4,1);
Cdagdn[order,order,1]=kron(sz,sp);
Cdagup=TensorMap(Cdagup,  V ← V ⊗Vdummy);Cdagup=permute(Cdagup,(3,1,),(2,))
Cdagdn=TensorMap(Cdagdn,  V ← V ⊗Vdummy);Cdagdn=permute(Cdagdn,(3,1,),(2,))

Cup=zeros(1,4,4);
Cup[1,order,order]=kron(sm,Id);
Cdn=zeros(1,4,4);
Cdn[1,order,order]=kron(sz,sm);
Cup=TensorMap(Cup, Vdummy ⊗ V ← V);
Cdn=TensorMap(Cdn, Vdummy ⊗ V ← V);



###################




t1=parameters["t1"];
t2=parameters["t2"];
ϕ=parameters["ϕ"];
μ=parameters["μ"];

ex_set=zeros(Lx,Ly)*im;
ey_set=zeros(Lx,Ly)*im;
e_right_bot_set=zeros(Lx,Ly)*im;
e0_set=zeros(Lx,Ly)*im;


E_total=0;



cx=1;cy=1;
ex=hopping_x(CTM_cell,Cdagup,Cup,A_cell,AA_cell,cx,cy,grad_ctm_setting);
ey=hopping_y(CTM_cell,Cdagup,Cup,A_cell,AA_cell,cx,cy,grad_ctm_setting);
e_right_bot=hopping_right_bot(CTM_cell,Cdagup,Cup,A_cell,AA_cell,cx,cy,grad_ctm_setting);
e0=ob_onsite(CTM_cell,N_occu,A_cell,AA_cell,cx,cy,grad_ctm_setting);

println("ex= "*string(ex))
println("ey= "*string(ey))
println("e_right_bot= "*string(e_right_bot))



cx=1;cy=1;
ex=hopping_x(CTM_cell,Cdagdn,Cdn,A_cell,AA_cell,cx,cy,grad_ctm_setting);
ey=hopping_y(CTM_cell,Cdagdn,Cdn,A_cell,AA_cell,cx,cy,grad_ctm_setting);
e_right_bot=hopping_right_bot(CTM_cell,Cdagdn,Cdn,A_cell,AA_cell,cx,cy,grad_ctm_setting);
e0=ob_onsite(CTM_cell,N_occu,A_cell,AA_cell,cx,cy,grad_ctm_setting);

println("ex= "*string(ex))
println("ey= "*string(ey))
println("e_right_bot= "*string(e_right_bot))

