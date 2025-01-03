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

include("../../src/bosonic/Settings.jl")
include("../../src/bosonic/Settings_cell.jl")
include("../../src/bosonic/iPEPS_ansatz.jl")
include("../../src/bosonic/AD_lib.jl")
include("../../src/bosonic/line_search_lib.jl")
include("../../src/bosonic/line_search_lib_cell.jl")
include("../../src/bosonic/optimkit_lib.jl")
include("../../src/bosonic/CTMRG.jl")
include("../../src/fermionic/Fermionic_CTMRG.jl")
include("../../src/fermionic/Fermionic_CTMRG_unitcell.jl")
include("../../src/fermionic/square_Hubbard_model_cell.jl")
include("../../src/fermionic/swap_funs.jl")
include("../../src/fermionic/fermi_permute.jl")
include("../../src/fermionic/mpo_mps_funs.jl")
include("../../src/fermionic/double_layer_funs.jl")
include("../../src/fermionic/square_Hubbard_AD_cell.jl")
include("../../src/fermionic/triangle_fiPESS_method.jl")
include("../../src/fermionic/simple_update/fermi_triangle_SimpleUpdate_iPESS.jl")
Random.seed!(888)

D=4;
chi=40

t1=1;
t2=1;
μ=0;
U=10;
B=0;
Chi_up_triangle=0;
Chi_dn_triangle=0;
@show parameters=Dict([("t1", t1),("t2", t2), ("μ",  μ), ("U",  U), ("B",  B), ("Chi_up_triangle", Chi_up_triangle), ("Chi_dn_triangle", Chi_dn_triangle)]);



import LinearAlgebra.BLAS as BLAS
n_cpu=10;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C"*string(n_cpu)*"_stoch_"*"iPESS_U"*string(U)*"_D"*string(D))
pid=getpid();
println("pid="*string(pid));

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
optim_setting.init_statenm="iPESS_D_4_chi_80.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "standard_triangle_Hubbard";
dump(energy_setting);

algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings


global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings


global Lx,Ly
Lx=1;
Ly=1;







V=Rep[SU₂](0=>2, 1/2=>1);
Vp=Rep[SU₂](0=>2, 1/2=>1);
B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS(Lx,Ly,V,Vp); 


global save_filenm
save_filenm="stochastic_iPESS_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"

global starting_time
starting_time=now();

################################################

B_set,T_set=to_C3_symmetric_fiPESS(B_set,T_set,0);

state_vec=Matrix{Triangle_iPESS}(undef,Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        state_vec[ca,cb]=Triangle_iPESS(T_set[ca,cb],B_set[ca,cb]);
        iPESS_to_iPEPS(state_vec[ca,cb]);
    end
end
###################################################





global E_history,E_all_history,delta_history
E_history=[10000];
E_all_history=[10000];
delta_history=[10000];

# maxiter=100;
# gtol=1e-3;
# delta=1e-3;
# state_vec=stochastic_opt(state_vec, delta, maxiter, gtol);

x=state_vec;
E_tem,∂E,CTM_tem=get_grad(x);


println(E_tem)

B_set_grad=Matrix{TensorMap}(undef,Lx,Ly)
T_set_grad=Matrix{TensorMap}(undef,Lx,Ly)
for ca=1:Lx
    for cb=1:Ly
        B_set_grad[ca,cb]=∂E[ca,cb].Tm;
        T_set_grad[ca,cb]=∂E[ca,cb].Bm;
    end
end

B_set_grad_,T_set_grad_=to_C3_symmetric_fiPESS(B_set_grad,T_set_grad,0);
B_set_grad_=B_set_grad_/3;
T_set_grad_=T_set_grad_/3;

@show dot(B_set_grad_,B_set_grad)/sqrt(dot(B_set_grad_,B_set_grad_)*dot(B_set_grad,B_set_grad))
@show dot(T_set_grad_,T_set_grad)/sqrt(dot(T_set_grad_,T_set_grad_)*dot(T_set_grad,T_set_grad))

# ls = BackTracking(order=3)
# println(ls)
# fx_bt3, x_bt3, iter_bt3 = gdoptimize(f, g!, fg!, state_vec, ls)