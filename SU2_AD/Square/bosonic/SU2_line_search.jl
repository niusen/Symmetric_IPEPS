using Revise, TensorKit, Zygote
using LinearAlgebra:I,diagm
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using LineSearches
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)

include("..\\..\\src\\bosonic\\square\\square_spin_operator.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\bosonic\\square\\square_model.jl")
include("..\\..\\src\\bosonic\\square\\square_AD_SU2.jl")
include("..\\..\\src\\bosonic\\Settings.jl")

include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
# include("..\\..\\src\\square_RVB_ansatz.jl")


Random.seed!(555)


D=3;
chi=54;


J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;


parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);


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
optim_setting.init_statenm="D3.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_Energy_settings();
energy_setting.model = "triangle_J1_J2_Jchi";
dump(energy_setting);





global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings






global Vv

if D==3
    Vv=SU2Space(0=>1,1/2=>1);
elseif D==4
    Vv=SU2Space(0=>2,1/2=>1);
elseif D==5
    Vv=SU2Space(0=>1,1/2=>2);
elseif D==6
    Vv=SU2Space(0=>1,1/2=>1,1=>1);
elseif D==8
    Vv=SU2Space(0=>1,1/2=>2,1=>1);
elseif D==11
    Vv=SU2Space(0=>1,1/2=>2,1=>2);
elseif D==16
    Vv=SU2Space(0=>1,1/2=>3,1=>3);    
end
@assert dim(Vv)==D;

global starting_time
starting_time=now();


#E_tem,∂E,CTM_tem=get_grad((triangle_tensor,triangle_tensor,bond_tensor,bond_tensor,bond_tensor));
#run_FiniteDiff(parameters, Vv, chi, LS_ctm_setting, optim_setting, energy_setting)

#fun(state_vec)
# global E_tem, CTM_tem
# E,∂E,CTM_tem=get_grad(state_vec);
# println(E,∂E)


# E0, grad=FD(state_vec)
# println(grad)
# println(∂E./grad)

init_complex_tensor=true;
init_C4_symetry=false;

state_vec=initial_SU2_state(Vv, optim_setting.init_statenm, optim_setting.init_noise,init_complex_tensor,init_C4_symetry)
state_vec=normalize_ansatz(state_vec);

# E0_, grad,CTM_tem=get_grad(state_vec);
# include("src\\kagome_AD_SU2.jl")
# E0,grad_=FD(state_vec)

global E_history
E_history=[10000];

global save_filenm;
save_filenm="SU2_D"*string(D)*".jld2";

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