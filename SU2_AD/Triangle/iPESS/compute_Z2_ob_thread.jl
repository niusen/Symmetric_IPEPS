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
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell_thread.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate_iPESS.jl")


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
    
    
    
    t1=1;
    t2=1;
    ϕ=pi/2;
    μ=0;
    U=15;
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
    LS_ctm_setting.projector_strategy="4x2";#"4x4" or "4x2"
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
    
    
    chi_set=[40 80 120 160];
    
    A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    CTM_cell=nothing;
    
    
    for cc in eachindex(chi_set)
        chi=chi_set[cc];
        @show chi
    
    
        CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, CTM_cell,LS_ctm_setting);
        E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        println(E_total)
        println(ex_set)
        println(ey_set)
        println(e_diagonala_set)
        println(e0_set)
        println(eU_set)
    
        if isa(space(A_cell_iPEPS[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
            sx_set,sy_set,sz_set=evaluate_spin_cell(A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting);
            S2=sqrt.(sx_set.^2+sy_set.^2+sz_set.^2);
            println("S2= "*string(abs.(S2))*", sx= "*string(sx_set)*", sy= "*string(sy_set)*", sz= "*string(sz_set));
        end
    
    
        filenm_="ob_SU_iPESS_Z2_D"*string(D_max_)*"_chi"*string(chi);
        matwrite(filenm_*".mat", Dict(
            "E_total" => E_total,
            "ex_set" => ex_set,
            "ey_set" => ey_set,
            "e_diagonala_set" => e_diagonala_set,
            "e0_set"=> e0_set,
            "eU_set" => eU_set,
            "sx_set" => sx_set,
            "sy_set" => sy_set,
            "sz_set" => sz_set,
            "S2" => S2
        ); compress = false)
    
    
        # jldsave(filenm_*".jld2";CTM_cell)
    
    
        init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
    
    end
    
    
    
end
    