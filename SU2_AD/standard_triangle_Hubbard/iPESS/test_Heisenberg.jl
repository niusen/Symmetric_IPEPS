using Distributed
using Revise, TensorKit, Zygote
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)



include("../../src/bosonic/Settings.jl")
include("../../src/bosonic/Settings_cell.jl")
include("../../src/bosonic/iPEPS_ansatz.jl")
include("../../src/bosonic/AD_lib.jl")
include("../../src/bosonic/line_search_lib.jl")
include("../../src/bosonic/line_search_lib_cell.jl")
include("../../src/bosonic/stochastic_opt.jl")
include("../../src/bosonic/optimkit_lib.jl")
include("../../src/bosonic/CTMRG.jl")
include("../../src/fermionic/Fermionic_CTMRG.jl")
include("../../src/fermionic/Fermionic_CTMRG_unitcell.jl")
include("../../src/fermionic/square_Hubbard_model_cell.jl")
include("../../src/fermionic/swap_funs.jl")
include("../../src/fermionic/mpo_mps_funs.jl")
include("../../src/fermionic/double_layer_funs.jl")
include("../../src/fermionic/square_Hubbard_AD_cell.jl")
include("../../src/fermionic/triangle_fiPESS_method.jl")
include("../../src/fermionic/triangle_fiPESS_method.jl")
include("../../src/fermionic/simple_update/fermi_triangle_SimpleUpdate.jl")
include("../../src/fermionic/simple_update/fermi_triangle_SimpleUpdate_iPESS.jl")

Random.seed!(666)

D=4;
chi=40

t1=0;
t2=0;
μ=0;
U=15;
B=0;
J=1.0;
Chi_up_triangle=0;
Chi_dn_triangle=0;
@show parameters=Dict([("t1", t1),("t2", t2), ("μ",  μ), ("U",  U), ("B",  B), ("J",  J), ("Chi_up_triangle", Chi_up_triangle), ("Chi_dn_triangle", Chi_dn_triangle)]);

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
optim_setting.init_statenm="stochastic_iPESS_Bfield_D_4_chi_40_S_0.2.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "standard_triangle_Hubbard_spiral";
dump(energy_setting);

algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings


global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings


###########################
import LinearAlgebra.BLAS as BLAS
@show BLAS_thread=6;
BLAS.set_num_threads(BLAS_thread);
Base.Sys.set_process_title("C"*string(BLAS_thread)*"_Z2_U"*string(U)*"_D"*string(D))
pid=getpid();
println("pid="*string(pid));
###########################


global Lx,Ly
Lx=1;
Ly=1;


Vspace=Rep[ℤ₂](0=>2,1=>2);



init_complex_tensor=true;

state_vec=initial_fiPESS_spinful_Z2(Vspace,optim_setting.init_statenm, optim_setting.init_noise,init_complex_tensor)
state_vec=normalize_ansatz(state_vec);


global save_filenm
save_filenm="stochastic_iPESS_spiral_D_"*string(D)*"_chi_"*string(chi)*".jld2"

global starting_time
starting_time=now();

################################################



global E_history,E_all_history,delta_history
E_history=[10000];
E_all_history=[10000];
delta_history=[10000];

maxiter=100;
gtol=1e-3;
delta=1e-3;
# state_vec=stochastic_opt(state_vec, delta, maxiter, gtol);

x=state_vec;

state=state_vec;
B_set=Matrix{TensorMap}(undef,Lx,Ly);
T_set=Matrix{TensorMap}(undef,Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        B_set[ca,cb]=state[ca,cb].Tm;
        T_set[ca,cb]=state[ca,cb].Bm;
        # state_new[ca,cb]=Triangle_iPESS(Tset[ca,cb],Bset[ca,cb]);
        # iPESS_to_iPEPS(state_new[ca,cb]);
    end
end

E_tem=0;

A_cell=convert_iPESS_to_iPEPS(B_set,T_set);
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);

E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, SSx_set, SSy_set, SSdiagonal_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM_cell); 
println("ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
println("E= "*string(E));flush(stdout);
println("ex_set= "*string(ex_set[:])); flush(stdout);
println("ey_set= "*string(ey_set[:]));flush(stdout);
println("e_diagonal1_set= "*string(e_diagonal1_set[:]));flush(stdout);
println("e0_set= "*string(e0_set[:]));flush(stdout);
println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
println("eU_set= "*string(eU_set[:])); flush(stdout);
println("SSx_set= "*string(SSx_set[:])); flush(stdout);
println("SSy_set= "*string(SSy_set[:])); flush(stdout);
println("SSdiagonal_set= "*string(SSdiagonal_set[:])); flush(stdout);

println(E_tem)