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

include("../../../../src/bosonic/Settings.jl")
include("../../../../src/bosonic/Settings_cell.jl")
include("../../../../src/bosonic/iPEPS_ansatz.jl")
include("../../../../src/bosonic/AD_lib.jl")
include("../../../../src/bosonic/line_search_lib.jl")
include("../../../../src/bosonic/line_search_lib_cell.jl")
include("../../../../src/bosonic/optimkit_lib.jl")
include("../../../../src/bosonic/CTMRG.jl")
include("../../../../src/fermionic/Fermionic_CTMRG.jl")
include("../../../../src/fermionic/Fermionic_CTMRG_unitcell.jl")
include("../../../../src/fermionic/square_Hubbard_model_cell.jl")
include("../../../../src/fermionic/swap_funs.jl")
include("../../../../src/fermionic/mpo_mps_funs.jl")
include("../../../../src/fermionic/double_layer_funs.jl")
include("../../../../src/fermionic/square_Hubbard_AD_cell.jl")
include("../../../../src/fermionic/triangle_fiPESS_method.jl")
Random.seed!(888)

D=4;
chi=40

t=1;
ϕ=pi/2;
μ=0;
U=0;
mx=0;
B=0;
mx_type="uniform";#"uniform" or "x_stagger"
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ), ("U",  U), ("B",  B), ("mx", mx), ("mx_type", mx_type)]);

###########################
import LinearAlgebra.BLAS as BLAS
n_cpu=10;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C"*string(n_cpu)*"_var_"*"U"*string(U)*"_D"*string(D))
pid=getpid();
println("pid="*string(pid));
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
###########################

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
optim_setting.init_statenm="nothing";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Triangle_Hofstadter_Hubbard_settings();
energy_setting.model = "Triangle_Hofstadter_Hubbard_spinHall";
energy_setting.Lx =2;
energy_setting.Ly =2;
energy_setting.Magnetic_cell =2;
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
Lx=2;
Ly=2;


if optim_setting.init_statenm=="nothing"
    Vp=Rep[ℤ₂](0=>2,1=>2);
    V=Rep[ℤ₂](0=>2,1=>2);
    init_complex_tensor=true;
    state_vec=initial_fiPESS_spinful_Z2(optim_setting.init_statenm,optim_setting.init_noise,init_complex_tensor; V=V);
else
    data=load(optim_setting.init_statenm);
    if haskey(data,"x")
        init_complex_tensor=true;
        state_vec=initial_fiPESS_spinful_Z2(optim_setting.init_statenm,optim_setting.init_noise,init_complex_tensor);
    else
        Tset=data["T_set"];
        Bset=data["B_set"];
        Lx,Ly=size(Tset);

        state_vec=Matrix{Triangle_iPESS}(undef,Lx,Ly);
        for ca=1:Lx
            for cb=1:Ly
                state_vec[ca,cb]=Triangle_iPESS(Tset[ca,cb],Bset[ca,cb]);
                iPESS_to_iPEPS(state_vec[ca,cb]);
            end
        end
    end
end



state_vec=normalize_ansatz(state_vec);


global save_filenm
save_filenm="var_iPESS_Z2_"*(energy_setting.model)*"_D"*string(D)*"_chi_"*string(chi)*".jld2";


global starting_time
starting_time=now();

################################################



global E_history
E_history=[10000];


x=Matrix{Triangle_iPESS_immutable}(undef,Lx,Ly);

for cc in eachindex(x)
    x[cc]=Triangle_iPESS_convert(state_vec[cc]);#convert to immutable ansatz
end
x_opt, fx, gx, numfg, grad_history = optimkit_op(x);