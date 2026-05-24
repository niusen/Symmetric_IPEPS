using Revise, TensorKit, Zygote
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)




include("../../../../src/bosonic/Settings.jl")
include("../../../../src/bosonic/Settings_cell.jl")
include("../../../../src/bosonic/iPEPS_ansatz.jl")
include("../../../../src/bosonic/AD_lib.jl")
include("../../../../src/bosonic/line_search_lib.jl")
include("../../../../src/bosonic/line_search_lib_cell.jl")
include("../../../../src/bosonic/optimkit_lib.jl")
# include("../../../../src/bosonic/CTMRG.jl")
# include("../../../../src/fermionic/Fermionic_CTMRG.jl")
# include("../../../../src/fermionic/Fermionic_CTMRG_unitcell.jl")
include("../../../../src/fermionic/square_Hubbard_model_cell.jl")
include("../../../../src/fermionic/square_Hubbard_AD_cell.jl")
include("../../../../src/fermionic/swap_funs.jl")
include("../../../../src/fermionic/fermi_permute.jl")
include("../../../../src/fermionic/mpo_mps_funs.jl")
include("../../../../src/fermionic/double_layer_funs.jl")
include("../../../../src/fermionic/triangle_fiPESS_method.jl")
include("../../../../src/fermionic/simple_update/fermi_triangle_SimpleUpdate.jl")
include("../../../../src/fermionic/simple_update/fermi_triangle_SimpleUpdate_iPESS.jl")
include("../../../../src/fermionic/simple_update/fermi_triangle_FullUpdate_iPESS.jl")
include("../../../../src/fermionic/simple_update/Full_Update_lib.jl")
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


D_max=4;

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
Base.Sys.set_process_title("C"*string(n_cpu)*"_FU_"*"U"*string(U)*"_D"*string(D_max))
pid=getpid();
println("pid="*string(pid));
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
###########################


function main(D_max,parameters)

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
optim_setting.init_statenm="nothing";#"SU_iPESS_SU2_csl_D6_3.93877.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

ENV_ctm_setting=LS_CTMRG_settings();
ENV_ctm_setting.CTM_conv_tol=1e-6;
ENV_ctm_setting.CTM_ite_nums=50;
ENV_ctm_setting.CTM_trun_tol=1e-8;
ENV_ctm_setting.svd_lanczos_tol=1e-8;
ENV_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
ENV_ctm_setting.conv_check="singular_value";
ENV_ctm_setting.CTM_ite_info=true;
ENV_ctm_setting.CTM_conv_info=true;
ENV_ctm_setting.CTM_trun_svd=false;
ENV_ctm_setting.construct_double_layer=true;
ENV_ctm_setting.grad_checkpoint=false;
dump(ENV_ctm_setting);


energy_setting=Triangle_Hofstadter_Hubbard_settings();
energy_setting.model = "Triangle_Hofstadter_Hubbard_spinHall";
energy_setting.Lx =2;
energy_setting.Ly =1;
energy_setting.Magnetic_cell =2;
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
projector_trun_tol=ENV_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=energy_setting.Lx;
Ly=energy_setting.Ly;


ENV_ctm_setting.CTM_ite_nums=30;
##############


global Lx,Ly,A_cell
global chi, parameters, energy_setting, grad_ctm_setting


if optim_setting.init_statenm=="nothing"
    Vp=Rep[ℤ₂](0=>2,1=>2);
    V=Rep[ℤ₂](0=>D_max/2,1=>D_max/2);
    B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS(Lx,Ly,V,Vp); 
else
    data=load(optim_setting.init_statenm);
    if haskey(data,"T_set")
        T_set=data["T_set"];
        B_set=data["B_set"];
    else
        state=data["x"];
        Lx,Ly=size(state);
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
    end
    for ca=1:Lx
        for cb=1:Ly
            bm=B_set[ca,cb];
            bm=add_noise(bm,optim_setting.init_noise,true);
            B_set[ca,cb]=bm;

            tm=T_set[ca,cb];
            tm=add_noise(tm,optim_setting.init_noise,true);
            T_set[ca,cb]=tm;
        end
    end

end

filenm="SU_iPESS_Z2_"*(energy_setting.model)*"_D"*string(D_max)*".jld2";
jldsave(filenm; B_set, T_set)



end



main(D_max,parameters)




