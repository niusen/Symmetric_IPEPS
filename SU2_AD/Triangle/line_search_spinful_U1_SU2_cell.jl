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

Random.seed!(888)

VDummytype=2;
D=4;
chi=40

t=1;
ϕ=pi/2;
μ=0;
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ)]);



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
optim_setting.init_statenm="Optim_cell_LS_D_4_chi_40_2.36933.jld2";#"Optim_cell_LS_D_4_chi_40_2.140901.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "spinful_triangle_lattice";
dump(energy_setting);

algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings


global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings

Vphy=Rep[U₁ × SU₂]((0,0)=>1,(2,0)=>1,(1, 1/2)=>1);


global VDummy_set, Vv_set

if VDummytype==1
    VDummy1=Rep[U₁ × SU₂]((-1, 1/2)=>1);
    VDummy2=Rep[U₁ × SU₂]((-1, 1/2)=>1);
    VDummy_set=(VDummy1,VDummy2,);
elseif VDummytype==2
    VDummy1=Rep[U₁ × SU₂]((-2, 0)=>1);
    VDummy2=ProductSpace{GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}, 0}();
    VDummy_set=(VDummy1,VDummy2,);
end

if VDummytype==1
    if D==4
        Vv1=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1);
        Vv_set=((Vv1,Vv1',Vv1',Vv1,),(Vv1,Vv1',Vv1',Vv1,),);
    elseif D==5
    elseif D==6
    elseif D==10
    end
elseif VDummytype==2
    if D==4
        Vv1=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1);
        Vv3=Rep[U₁ × SU₂]((-1, 1/2)=>1, (0, 0)=>2, (1, 1/2)=>1)';
        Vv_set=((Vv1,Vv1',Vv3,Vv1,),(Vv3',Vv1',Vv1',Vv1,),);
    elseif D==5
    elseif D==6
    elseif D==10
    end
end



global Lx,Ly
Lx=2;
Ly=1;






init_complex_tensor=true;

state_vec=initial_fPEPS_state_spinful_U1_SU2(Vphy,Vv_set, optim_setting.init_statenm, optim_setting.init_noise,init_complex_tensor)
state_vec=normalize_tensor_group(state_vec);


global save_filenm
save_filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"

global starting_time
starting_time=now();

################################################



global E_history
E_history=[10000];


ls = BackTracking(order=3)
println(ls)
fx_bt3, x_bt3, iter_bt3 = gdoptimize(f, g!, fg!, state_vec, ls)

# ls = StrongWolfe()
# println(ls)
# fx_sw, x_sw, iter_sw = gdoptimize(f, g!, fg!, state_vec, ls)

# ls = LineSearches.HagerZhang()
# println(ls)
# fx_hz, x_hz, iter_hz = gdoptimize(f, g!, fg!, state_vec, ls)

# ls = MoreThuente()
# println(ls)
# fx_mt, x_mt, iter_mt = gdoptimize(f, g!, fg!, state_vec, ls)


# #optimize with OptimKit
# optimkit_op(state_vec)


println(E_tem)