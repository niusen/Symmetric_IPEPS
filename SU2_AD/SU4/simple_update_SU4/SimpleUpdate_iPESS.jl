using Distributed
using LinearAlgebra:I,diag,diagm,kron
using TensorKit,TensorKitSectors
using SUNRepresentations
using JLD2,MAT
using KrylovKit
using JSON
using Random
using ChainRulesCore
using Zygote
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)




include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\SUN_spin_op.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\bosonic\\CTMRG_unitcell.jl")
include("..\\..\\src\\bosonic\\triangle\\triangle_spin_model_cell.jl")
include("..\\..\\src\\bosonic\\triangle\\triangle_spin_AD_cell.jl")
include("..\\..\\src\\bosonic\\triangle\\triangle_iPESS_method.jl")
include("..\\..\\src\\bosonic\\triangle\\simple_update\\triangle_SimpleUpdate.jl")
include("..\\..\\src\\bosonic\\triangle\\simple_update\\triangle_SimpleUpdate_iPESS.jl")

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""


####################
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
import LinearAlgebra.BLAS as BLAS

if nworkers()==1
  n_cpu=10;
  BLAS.set_num_threads(n_cpu);
  println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);
else
    # @everywhere BLAS.set_num_threads(1)
    # BLAS.set_num_threads(8);#main processer
    # @show n_cpu=nworkers()#*(BLAS.get_num_threads());
end
Base.Sys.set_process_title("C"*string(n_cpu)*"_MPS")
main_pid=getpid();
println("pid="*string(main_pid));;flush(stdout);
#change name process for each worker
if nworkers()>1
for (i, pid) in enumerate(workers())
    remotecall_wait(() -> begin
        title = string(main_pid)*"-$i"
        Base.Sys.set_process_title(title)
        println("Worker $(myid()): set process title to '$title'")
    end, pid)
end
end

##########################
SUNRepresentations.cache_info()
#SUNRepresentations.precompute_disk_cache(N, a_max)
# SUNRepresentations.precompute_disk_cache(3)
#SUNRepresentations.CGC_CACHE_PATH
for mode in ["weight", "dynkin", "dimension"]
    SUNRepresentations.display_mode(mode)
    # @show SUNIrrep(2,2,2,0)
end
####################
# @everywhere cache_setting();
##########################

D_max=15;


J=1;
K=0;
Φ=0;
parameters=Dict([("J", J),("K", K), ("Φ",  Φ)]);



trun_tol=1e-6;



chi=200;

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
optim_setting.init_statenm="nothing";#"Gutzwiller_stochastic_iPESS_LS_D_6_chi_40_2.3592.jld2";#"nothing";
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

energy_setting=Triangle_SUN_Spin_settings();
energy_setting.model = "Heisenberg";
energy_setting.N=4;
energy_setting.Lx=2;
energy_setting.Ly=2;
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
Lx=energy_setting.Lx;
Ly=energy_setting.Ly;



##############


global Lx,Ly,A_cell
global chi, parameters, energy_setting, grad_ctm_setting


if optim_setting.init_statenm=="nothing"
    V=Rep[SU₄]("1"=>1, "4"=>1, "6"=>1, "4⁺"=>1);
    V=Rep[SU₄]("1"=>1, "4"=>1, "4⁺"=>1);
    Vp=Rep[SU₄]("4"=>1);
    B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS(Lx,Ly,V,Vp); 
    # B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS_uniform(Lx,Ly,V,Vp);    
else
    data=load(optim_setting.init_statenm);
    if haskey(data,"x")
        x=data["x"];
        Lx,Ly=size(x);
        B_set=Matrix{Any}(undef,Lx,Ly);
        T_set=Matrix{Any}(undef,Lx,Ly);
        λ_set1=Matrix{Any}(undef,Lx,Ly);
        λ_set2=Matrix{Any}(undef,Lx,Ly);
        λ_set3=Matrix{Any}(undef,Lx,Ly);
    
        for ca=1:Lx
            for cb=1:Ly
                bm=x[ca,cb].Tm;
                tm=x[ca,cb].Bm;
                T_set[ca,cb]=tm;
                B_set[ca,cb]=bm;
                t_A=bm;
                λ_A_1=unitary(space(t_A,1)',space(t_A,1)');
                λ_A_2=unitary(space(t_A,2)',space(t_A,2)');
                λ_A_3=unitary(space(t_A,3)',space(t_A,3)');
                λ_set1[ca,cb]=λ_A_1;
                λ_set2[ca,cb]=λ_A_2;
                λ_set3[ca,cb]=λ_A_3;
            end
        end
    else
        B_set=data["B_set"];
        T_set=data["T_set"];
        λ_set1=data["λ_set1"];
        λ_set2=data["λ_set2"];
        λ_set3=data["λ_set3"];
    end
end


B_set,T_set=to_C3_symmetric_iPESS(B_set,T_set);





# A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
# init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
# CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,LS_ctm_setting);
# E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
# println(E_total)
# println(ex_set)
# println(ey_set)
# println(e_diagonala_set)
# println(e0_set)
# println(eU_set)

D0set=[];
for cc in eachindex(B_set)
    D0set=vcat(D0set,[dim(space(B_set[cc],1)), dim(space(B_set[cc],2)), dim(space(B_set[cc],3))]);
end
D_max0=maximum(D0set);
B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS_no_Hamiltonian(parameters,energy_setting, B_set, T_set, λ_set1, λ_set2, λ_set3, D_max0, trun_tol);



point_group_projection


tau=20;
dt=0.1;
B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS(parameters,energy_setting, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);

# tau=20;
# dt=0.05;
# B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS(parameters,energy_setting, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);

tau=20;
dt=0.02;
B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS(parameters,energy_setting, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);



# tau=20;
# dt=0.002;
# B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS(parameters,energy_setting, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);



A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,LS_ctm_setting);
E_total,  ex_set, ey_set, e_diagonal_set, triangle_right_bot_set, triangle_left_top_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(E_total)
println(ex_set)
println(ey_set)
println(e_diagonal_set)
println(triangle_right_bot_set)
println(triangle_left_top_set)


filenm="SU_iPESS_SU2_D"*string(D_max)*".jld2";
jldsave(filenm; B_set, T_set, λ_set1, λ_set2, λ_set3)


# end
