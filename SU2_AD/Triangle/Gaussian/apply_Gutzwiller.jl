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

include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD.jl")

Random.seed!(888)

M=2;
Dx=16;
Dy=16;
chi=40

t=1;
ϕ=pi/2;
μ=0;
U=12;
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);




grad_ctm_setting=grad_CTMRG_settings();
grad_ctm_setting.CTM_conv_tol=1e-6;
grad_ctm_setting.CTM_ite_nums=10;
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
LS_ctm_setting.CTM_ite_nums=50;
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
optim_setting.init_statenm="parton_state_M2.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "spinful_triangle_lattice_2site";
dump(energy_setting);




global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings
global Vx,Vy

if (Dx==4)&(Dy==4)
    Vx=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)';
    Vy=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1);
elseif (Dx==4)&(Dy==8)
    Vx=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)';
    Vy=Rep[U₁ × SU₂]((0, 0)=>2, (2, 0)=>2, (1, 1/2)=>2);
 
elseif (Dx==16)&(Dy==16)
    Vx=Rep[U₁ × SU₂]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1);
    Vy=Rep[U₁ × SU₂]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1)';
end
@assert dim(Vx)==Dx;
@assert dim(Vy)==Dy;




init_complex_tensor=true;

state_vec=initial_fPEPS_spinful_U1_SU2_2site(Vx,Vy, optim_setting.init_statenm, optim_setting.init_noise,init_complex_tensor)
state_vec=normalize_tensor_group(state_vec);


global chi, parameters, energy_setting, grad_ctm_setting




for cg=1:10
    coe=cg/10;
    PG_set=Gutzwiller_U1_SU2_2site(coe);

    @tensor A[:]:=state_vec.T[-1,-2,-3,-4,1]*PG_set[1][-5,2]*PG_set[2][2,1]
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    E, ex1,ex2,ey1,ey2,e_diagonalb1,e_diagonalb2, e01,e02,eU1,eU2, ite_num,ite_err,CTM=energy_CTM(Square_iPEPS(A), chi, parameters, LS_ctm_setting, energy_setting, init, init_CTM);
    println([coe, E]);flush(stdout);
end





coe=0.5;
PG_set=Gutzwiller_U1_SU2_2site(coe);
@tensor A[:]:=state_vec.T[-1,-2,-3,-4,1]*PG_set[1][-5,2]*PG_set[2][2,1]
filenm="Gutzwiller_M"*string(M)*"_coe_"*string(coe)*".jld2"
jldsave(filenm;A=A)