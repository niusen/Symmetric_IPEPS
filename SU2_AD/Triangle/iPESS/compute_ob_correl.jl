using Revise
using LinearAlgebra:diag,I,diagm 
using TensorKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
using Random
# using LineSearches,OptimKit
using Dates
cd(@__DIR__)


include("../../../src/bosonic/Settings.jl")
include("../../../src/bosonic/Settings_cell.jl")
include("../../../src/bosonic/iPEPS_ansatz.jl")
include("../../../src/bosonic/AD_lib.jl")
include("../../../src/bosonic/line_search_lib.jl")
include("../../../src/bosonic/line_search_lib_cell.jl")
include("../../../src/bosonic/stochastic_opt.jl")
include("../../../src/bosonic/optimkit_lib.jl")
include("../../../src/bosonic/CTMRG.jl")
include("../../../src/fermionic/Fermionic_CTMRG.jl")
include("../../../src/fermionic/Fermionic_CTMRG_unitcell.jl")
include("../../../src/fermionic/square_Hubbard_model_cell.jl")
include("../../../src/fermionic/swap_funs.jl")
include("../../../src/fermionic/mpo_mps_funs.jl")
include("../../../src/fermionic/double_layer_funs.jl")
include("../../../src/fermionic/square_Hubbard_AD_cell.jl")
include("../../../src/fermionic/triangle_fiPESS_method.jl")
include("../../../src/fermionic/simple_update/fermi_triangle_SimpleUpdate.jl")
include("../../../src/fermionic/simple_update/fermi_triangle_SimpleUpdate_iPESS.jl")
include("../../../src/fermionic/square_Hubbard_model_correl_cell.jl")

begin
t=1;
ϕ=pi/2;
μ=0;
U=14;
B=0;
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ), ("U",  U), ("B",  B)]);

import LinearAlgebra.BLAS as BLAS
n_cpu=6;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C"*string(n_cpu)*"_"*"correl_U"*string(U))
pid=getpid();
println("pid="*string(pid));

LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=50;
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

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);

optim_setting=Optim_settings();
optim_setting.init_statenm="newnewversion_FU_iPESS_LS_D_12_chi_80.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
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
projector_trun_tol=LS_ctm_setting.CTM_trun_tol

global backward_settings





global Lx,Ly

if optim_setting.init_statenm=="nothing"
    # V=Rep[SU₂](0=>2, 1/2=>1);
    # Vp=Rep[SU₂](0=>2, 1/2=>1);
    # B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS(Lx,Ly,V,Vp); 
    # # B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS_uniform(Lx,Ly,V,Vp);    
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

    # λ_set1=data["λ_set1"];
    # λ_set2=data["λ_set2"];
    # λ_set3=data["λ_set3"];
end


Lx,Ly=size(B_set);



global chi, parameters, energy_setting, grad_ctm_setting

######################
A_cell=convert_iPESS_to_iPEPS(B_set,T_set);

###############

###############

chis=[40,80,120,160];

for cchi=1:length(chis)
    global chi,D
    D=dim(space(A_cell[1][1],1));
    @show chi=chis[cchi];
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);
    #E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
    E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
    # println(E_total)
    # println(ex_set)
    # println(ey_set)
    # println(e_diagonala_set)
    # println(e0_set)
    # println(eU_set)
    println("E= "*string(E_total));flush(stdout);
    println("ex_set= "*string(ex_set[:])); flush(stdout);
    println("ey_set= "*string(ey_set[:]));flush(stdout);
    println("e_diagonala_set= "*string(e_diagonala_set[:]));flush(stdout);
    println("e0_set= "*string(e0_set[:]));flush(stdout);
    println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
    println("eU_set= "*string(eU_set[:])); flush(stdout);

    triangle_up_set,triangle_dn_set,SS_x_set,SS_y_set,SS_diagonal_set=evaluate_spin_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);



    filenm_="ob_D"*string(D)*"_chi"*string(chi);
    matwrite(filenm_*".mat", Dict(
        "E_total" => E_total,
        "ex_set" => ex_set,
        "ey_set" => ey_set,
        "e_diagonala_set" => e_diagonala_set,
        "e0_set"=> e0_set,
        "eU_set" => eU_set,
        "triangle_up_set" =>triangle_up_set,
        "triangle_dn_set" =>triangle_dn_set,
        "SS_x_set"=>SS_x_set,
        "SS_y_set"=>SS_y_set,
        "SS_diagonal_set"=>SS_diagonal_set
    ); compress = false)


    distance=40;
    
    direction="x";
    SS_ob_set,CdagC_ob_set=cal_correl(CTM_cell,A_cell,AA_cell,D,chi,parameters,direction,distance);

end
#############################

end


