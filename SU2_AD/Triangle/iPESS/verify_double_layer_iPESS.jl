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
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell_iPESS.jl")
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
include("..\\..\\src\\fermionic\\verification_double_layer_iPESS.jl")


###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################

Random.seed!(888);



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
optim_setting.init_statenm="SU_iPESS_Z2_csl_D4.jld2";#"SU_iPESS_SU2_csl_D4.jld2";#"nothing";
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
    T_set=data["T_set"];
    B_set=data["B_set"];
    # λ_set1=data["λ_set1"];
    # λ_set2=data["λ_set2"];
    # λ_set3=data["λ_set3"];
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
Base.Sys.set_process_title("C"*string(n_cpu)*"_"*"ob_U"*string(U)*"_D"*string(D_max_))
pid=getpid();
println("pid="*string(pid));
###########################


##############################################################
A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
for c1=1:Lx;
    for c2=1:Ly;
        @time begin
            B_LU=B_set[c1,c2];
            B_double_LU, U_L,U_M,U_U = build_double_layer_swap_Tm(B_LU',B_LU, false);#L M U
            T_LU=T_set[c1,c2];
            T_double_LU, U_D,U_R,U_M = build_double_layer_swap_Bm(T_LU',T_LU, true);#D R M
            @tensor AA[:]:=B_double_LU[-1,1,-4]*T_double_LU[-2,-3,1];
        end

        AA0,_=build_double_layer_swap(A_cell_iPEPS[c1][c2]',A_cell_iPEPS[c1][c2]);

        @assert norm(AA-AA0)<1e-12;
        @show norm(AA-AA0);
    end
end



##########################################
LS_ctm_setting.CTM_ite_nums=2;
A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell=nothing;
chi=40;
_, AA_cell, _=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, CTM_cell,LS_ctm_setting);
CTM_cell, double_B_cell,double_T_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(B_set,T_set, A_cell_iPEPS,chi,init, CTM_cell,LS_ctm_setting);

for cx=1:Lx;
    for cy=1:Ly;
        verify_hopping_diagonala_iPESS(CTM_cell,A_cell_iPEPS,AA_cell, B_set,T_set, double_B_cell, double_T_cell, cx,cy,LS_ctm_setting,energy_setting);
        verify_hopping_y_iPESS(CTM_cell,A_cell_iPEPS,AA_cell, B_set,T_set, double_B_cell, double_T_cell, cx,cy,LS_ctm_setting,energy_setting);
        verify_hopping_x_iPESS(CTM_cell,A_cell_iPEPS,AA_cell, B_set,T_set, double_B_cell, double_T_cell, cx,cy,LS_ctm_setting,energy_setting);
    end
end
