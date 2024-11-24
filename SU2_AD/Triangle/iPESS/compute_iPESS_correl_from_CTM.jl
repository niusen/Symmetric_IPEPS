using Distributed
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
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell_iPESS_speed.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate_iPESS.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_FullUpdate_iPESS.jl")
include("..\\..\\src\\fermionic\\simple_update\\Full_Update_lib.jl")
include("..\\..\\src\\fermionic\\fermion_ob_iPESS.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_iPESS_correl_cell.jl")


###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################
let
Random.seed!(888);
@show workers()


t1=1;
t2=1;
ϕ=pi/2;
μ=0;
U=12;
B=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("U",  U), ("B",  B)]);




trun_tol=1e-6;



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

optim_setting=Optim_settings();
optim_setting.init_statenm="SU_iPESS_Z2_csl_D6.jld2";#"SU_iPESS_SU2_csl_D4.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);


LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=10;
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


##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=6;
Ly=6;
@assert mod(Lx,2)==0; #even unitcell in x direction due to the flux


LS_ctm_setting.CTM_ite_nums=30;
##############


global Lx,Ly,A_cell
global chi, parameters, energy_setting, grad_ctm_setting


if optim_setting.init_statenm=="nothing"
    Vp=Rep[ℤ₂](0=>2,1=>2);
    V=Rep[ℤ₂](0=>1,1=>1);
    B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS(Lx,Ly,V,Vp);    
else
    data=load(optim_setting.init_statenm);
    if haskey(data,"x")
        x=data["x"];
        @assert (Lx,Ly)==size(x);
        T_set=Matrix{TensorMap}(undef,Lx,Ly);
        B_set=Matrix{TensorMap}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                T_set[cx,cy]=x[cx,cy].Bm;
                B_set[cx,cy]=x[cx,cy].Tm;
                # A=iPESS_to_iPEPS(x[cx,cy]);
                # A=A.T;
                # AA_,_=build_double_layer_swap(A',A);
                # println(norm(AA_))
            end
        end
    else
        T_set=data["T_set"];
        B_set=data["B_set"];
        @assert (Lx,Ly)==size(B_set);
        # λ_set1=data["λ_set1"];
        # λ_set2=data["λ_set2"];
        # λ_set3=data["λ_set3"];
    end
end


D_max_=0;
for cx=1:Lx
    for cy=1:Ly
        sp1=dim(space(B_set[cx,cy],1));
        sp2=dim(space(B_set[cx,cy],2));
        sp3=dim(space(B_set[cx,cy],1));
        D_max_=maximum([D_max_ sp1 sp2 sp3]);
    end
end



###########################
import LinearAlgebra.BLAS as BLAS
n_cpu=6;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))
Base.Sys.set_process_title("corr"*"_U"*string(U)*"_D"*string(D_max_))
pid=getpid();
println("pid="*string(pid));
###########################


chi_set=[40 80 120 160];

A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell=nothing;



##########################
double_B_cell=Matrix{TensorMap}(undef,Lx,Ly);
double_T_cell=Matrix{TensorMap}(undef,Lx,Ly);
# U_L_cell=Matrix{TensorMap}(undef,Lx,Ly);
# U_D_cell=Matrix{TensorMap}(undef,Lx,Ly);
# U_R_cell=Matrix{TensorMap}(undef,Lx,Ly);
# U_U_cell=Matrix{TensorMap}(undef,Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        AA_, T_double, B_double, U_L_,U_D_,U_R_,U_U_=build_doublelayer_swap_iPESS(B_set,T_set, [cx,cy]);
        # println(norm(AA_))
        double_B_cell[cx,cy]=B_double;
        double_T_cell[cx,cy]=T_double;
        # U_L_cell[cx,cy]=U_L_cell;
        # U_D_cell[cx,cy]=U_D_cell;
        # U_R_cell[cx,cy]=U_R_cell;
        # U_U_cell[cx,cy]=U_U_cell;
    end
end
##########################


for cc in eachindex(chi_set)
    chi=chi_set[cc];
    @show chi;flush(stdout);

    jldname="CTM_iPESS_D"*string(D_max_)*"_chi"*string(chi)*".jld2";
    data=load(jldname);
    CTM_cell=data["CTM_cell"];

    distance=40;
    direction="x";
    partly=true;
    SS_ob_set,CdagC_ob_set=cal_correl(CTM_cell,B_set,T_set,double_B_cell, double_T_cell,D_max_,chi,parameters,direction,distance,partly);




end



end
