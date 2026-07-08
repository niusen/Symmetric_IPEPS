using Revise
using LinearAlgebra:diag,I,diagm 
using TensorKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
using Random
using LineSearches
using KrylovKit
using Dates
cd(@__DIR__)

@show run_device="cuda:1"; # choose from "cpu", "cuda:0", "cuda:1"
@show ctm_device=run_device;
@show observable_device=run_device;
@show memory_info=false; # print tensor/GPU memory diagnostics
if any(dev -> lowercase(strip(dev)) != "cpu", (run_device, ctm_device, observable_device))
    using CUDA, cuTENSOR, Adapt
end

const SRC_ROOT=normpath(joinpath(@__DIR__,"..","..","src"))

include(joinpath(SRC_ROOT,"tensorkit_compat.jl"))
include(joinpath(SRC_ROOT,"bosonic","Settings.jl"))
include(joinpath(SRC_ROOT,"bosonic","Settings_cell.jl"))
include(joinpath(SRC_ROOT,"device_utils.jl"))
include(joinpath(SRC_ROOT,"bosonic","iPEPS_ansatz.jl"))
include(joinpath(SRC_ROOT,"bosonic","AD_lib.jl"))
include(joinpath(SRC_ROOT,"bosonic","line_search_lib.jl"))
include(joinpath(SRC_ROOT,"bosonic","line_search_lib_cell.jl"))
include(joinpath(SRC_ROOT,"bosonic","stochastic_opt.jl"))
include(joinpath(SRC_ROOT,"bosonic","optimkit_lib.jl"))
include(joinpath(SRC_ROOT,"bosonic","CTMRG.jl"))
include(joinpath(SRC_ROOT,"fermionic","Fermionic_CTMRG.jl"))
include(joinpath(SRC_ROOT,"fermionic","Fermionic_CTMRG_unitcell_iPESS.jl"))
include(joinpath(SRC_ROOT,"fermionic","square_Hubbard_model_cell.jl"))
include(joinpath(SRC_ROOT,"fermionic","swap_funs.jl"))
include(joinpath(SRC_ROOT,"fermionic","fermi_permute.jl"))
include(joinpath(SRC_ROOT,"fermionic","mpo_mps_funs.jl"))
include(joinpath(SRC_ROOT,"fermionic","double_layer_funs.jl"))
include(joinpath(SRC_ROOT,"fermionic","square_Hubbard_AD_cell.jl"))
include(joinpath(SRC_ROOT,"fermionic","triangle_fiPESS_method.jl"))
include(joinpath(SRC_ROOT,"fermionic","simple_update","fermi_triangle_SimpleUpdate.jl"))
include(joinpath(SRC_ROOT,"fermionic","simple_update","fermi_triangle_SimpleUpdate_iPESS.jl"))
include(joinpath(SRC_ROOT,"fermionic","simple_update","fermi_triangle_FullUpdate_iPESS.jl"))
include(joinpath(SRC_ROOT,"fermionic","triangle_Hubbard_model_cell.jl"))
include(joinpath(SRC_ROOT,"fermionic","square_Hubbard_model_iPESS_correl_cell.jl"))
include(joinpath(SRC_ROOT,"fermionic","square_Hubbard_model_correl_cell.jl"))

function _compute_as_tuple_cell(X, Lx, Ly)
    X isa Tuple && return X
    X_tuple=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            X_tuple=fill_tuple(X_tuple, X[cx,cy], cx,cy);
        end
    end
    return X_tuple
end

function _compute_as_matrix_cell(X, Lx, Ly)
    X isa AbstractMatrix && return X
    X_mat=Matrix{Any}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            X_mat[cx,cy]=X[cx][cy];
        end
    end
    return X_mat
end

begin
t=1;
ϕ=pi/2;
μ=-2;
U=9;
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
optim_setting.init_statenm="FU_iPESS_LS_D_14_chi_80_-1.03412.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
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

ipeps_select_device!(run_device)
ipeps_set_step_devices!(
    ctm=ctm_device,
    full_update=run_device,
    observable=observable_device,
)
ipeps_set_memory_info!(memory_info)

_same_device(dev1,dev2)=lowercase(strip(String(dev1)))==lowercase(strip(String(dev2)))


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
energy_setting.Lx=Lx;
energy_setting.Ly=Ly;

B_set=ipeps_to_device(IPESS_CTM_DEVICE[], B_set);
T_set=ipeps_to_device(IPESS_CTM_DEVICE[], T_set);
B_set=_compute_as_tuple_cell(B_set,Lx,Ly);
T_set=_compute_as_tuple_cell(T_set,Lx,Ly);



global chi, parameters, energy_setting, grad_ctm_setting

chis=[40,80,120,160];

for cchi=1:length(chis)
    global chi,D
    D=dim(space(B_set[1][1],1));
    @show chi=chis[cchi];
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    CTM_cell, double_B_cell, double_T_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(B_set,T_set,chi,init, init_CTM,LS_ctm_setting);
    U_L_cell=nothing;
    U_D_cell=nothing;
    U_R_cell=nothing;
    U_U_cell=nothing;
    ipeps_reclaim_device_memory!(aggressive=true);

    if _same_device(IPESS_OBSERVABLE_DEVICE[], IPESS_CTM_DEVICE[])
        B_ob=B_set;
        T_ob=T_set;
        double_B_ob=double_B_cell;
        double_T_ob=double_T_cell;
        CTM_ob=CTM_cell;
    else
        B_ob=ipeps_to_device(IPESS_OBSERVABLE_DEVICE[], B_set);
        T_ob=ipeps_to_device(IPESS_OBSERVABLE_DEVICE[], T_set);
        double_B_ob=ipeps_to_device(IPESS_OBSERVABLE_DEVICE[], double_B_cell);
        double_T_ob=ipeps_to_device(IPESS_OBSERVABLE_DEVICE[], double_T_cell);
        CTM_ob=ipeps_to_device(IPESS_OBSERVABLE_DEVICE[], CTM_cell);
    end
    E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell_iPESS(parameters, B_ob, T_ob, double_B_ob, double_T_ob, CTM_ob, LS_ctm_setting, energy_setting);
    ipeps_reclaim_device_memory!(aggressive=true);
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

    triangle_up_set,triangle_dn_set,SS_x_set,SS_y_set,SS_diagonal_set=evaluate_spin_ob_cell_iPESS(B_ob, T_ob, double_B_ob, double_T_ob, CTM_ob, LS_ctm_setting, energy_setting);
    ipeps_reclaim_device_memory!(aggressive=true);
    pairing_x_set, pairing_y_set, pairing_diagonala_set=evaluate_ob_pairing_cell_iPESS(parameters, B_ob, T_ob, double_B_ob, double_T_ob, CTM_ob, LS_ctm_setting, energy_setting);
    ipeps_reclaim_device_memory!(aggressive=true);



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
        "SS_diagonal_set"=>SS_diagonal_set,
        "pairing_x_set"=>pairing_x_set,
        "pairing_y_set"=>pairing_y_set,
        "pairing_diagonala_set"=>pairing_diagonala_set
    ); compress = false)


    distance=40;
    
    direction="x";
    partly=true;
    B_correl=_compute_as_matrix_cell(B_ob,Lx,Ly);
    T_correl=_compute_as_matrix_cell(T_ob,Lx,Ly);
    double_B_correl=_compute_as_matrix_cell(double_B_ob,Lx,Ly);
    double_T_correl=_compute_as_matrix_cell(double_T_ob,Lx,Ly);
    SS_ob_set,CdagC_ob_set=cal_correl(CTM_ob,B_correl,T_correl,double_B_correl,double_T_correl,D,chi,parameters,direction,distance,partly);
    B_ob=nothing;
    T_ob=nothing;
    double_B_ob=nothing;
    double_T_ob=nothing;
    B_correl=nothing;
    T_correl=nothing;
    double_B_correl=nothing;
    double_T_correl=nothing;
    CTM_ob=nothing;
    double_B_cell=nothing;
    double_T_cell=nothing;
    CTM_cell=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end

end
#############################

end


