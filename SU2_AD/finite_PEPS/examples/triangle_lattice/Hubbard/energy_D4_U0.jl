    using LinearAlgebra
    using TensorKit
    using KrylovKit
    using JSON
    using ChainRulesCore,Zygote
    using HDF5, JLD2, MAT
    using Random
    cd(@__DIR__)

    include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
    include("..\\..\\..\\..\\src\\fermionic\\swap_funs.jl")
    include("..\\..\\..\\..\\src\\fermionic\\fermi_permute.jl")
    include("..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
    include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
    include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
    include("..\\..\\..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")

    include("..\\..\\..\\setting\\Settings.jl")
    include("..\\..\\..\\setting\\tuple_methods.jl")
    include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
    include("..\\..\\..\\environment\\AD\\mps_methods.jl")
    include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
    include("..\\..\\..\\environment\\AD\\fermion\\peps_double_layer_methods_fermion.jl")
    include("..\\..\\..\\environment\\AD\\fermion\\fermi_CTM_observables.jl")
    include("..\\..\\..\\environment\\AD\\truncations.jl")
    include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
    include("..\\..\\..\\models\\Hubbard\\triangle_lattice\\Hofstadter.jl")

    D=4;


    global use_AD;
    use_AD=false;

    t1=1;
    t2=1;
    ϕ=pi/2;
    μ=0;
    U=0;
    parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);
    global parameters


    svd_settings=Svd_settings();
    svd_settings.svd_trun_method="chi";#chi" or "tol"
    svd_settings.chi_max=500;
    svd_settings.tol=1e-5;
    dump(svd_settings);

    backward_settings=Backward_settings();
    backward_settings.grad_inverse_tol=1e-8
    backward_settings.grad_regulation_epsilon=1e-12;
    backward_settings.show_ite_grad_norm=false;
    dump(backward_settings);
    global svd_settings, backward_settings



    #Hamiltonian
    # H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """

    Lx=4;
    Ly=4;

    data=load("FU_iPESS_LS_D_4_chi_40_2.17125.jld2");
    # data=load("SU_iPESS_SU2_csl_D6_2.259.jld2")
    T_virt_set=data["B_set"];
    T_phy_set=data["T_set"];



    #unit-cell of iPESS: 2x2
    iPESS_cell=[2,2];
    psi=Matrix{Triangle_iPESS}(undef,Lx,Ly);#PBC-PBC
    for cx=1:Lx
        for cy=1:Ly
            psi[cx,cy]=Triangle_iPESS(T_phy_set[mod1(cx,iPESS_cell[1]),mod1(cy,iPESS_cell[2])],T_virt_set[mod1(cx,iPESS_cell[1]),mod1(cy,iPESS_cell[2])]);
        end
    end

    #left boundary
    cx=1;
    for cy=1:Ly
        VL=Rep[SU₂](0=>1);
        V=space(psi[1,cy].Tm,1);
        iso=create_isometry(V,VL);
        T=psi[cx,cy].Tm;
        @tensor T[:]:=T[1,-2,-3]*iso'[-1,1];
        T=permute(T,(1,2,),(3,));
        psi[cx,cy].Tm=T;
    end

    #right boundary
    cx=Lx;
    for cy=1:Ly
        VL=Rep[SU₂](0=>1)';
        V=space(psi[cx,cy].Bm,3);
        iso=create_isometry(V,VL);
        T=psi[cx,cy].Bm;
        @tensor T[:]:=T[-1,-2,1,-4]*iso'[-3,1];
        T=permute(T,(1,),(2,3,4,));
        psi[cx,cy].Bm=T;
    end


    #bot boundary
    cy=1;
    for cx=1:Lx
        VL=Rep[SU₂](0=>1)';
        V=space(psi[cx,cy].Bm,4);
        iso=create_isometry(V,VL);
        T=psi[cx,cy].Bm;
        @tensor T[:]:=T[-1,-2,-3,1]*iso'[-4,1];
        T=permute(T,(1,),(2,3,4,));
        psi[cx,cy].Bm=T;
    end

    #top boundary
    cy=Ly;
    for cx=1:Lx
        VL=Rep[SU₂](0=>1);
        V=space(psi[cx,cy].Tm,2);
        iso=create_isometry(V,VL);
        T=psi[cx,cy].Tm;
        @tensor T[:]:=T[-1,1,-3]*iso'[-2,1];
        T=permute(T,(1,2,),(3,));
        psi[cx,cy].Tm=T;
    end



    psi_PEPS=iPESS_to_iPEPS_matrix(psi);


    # psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

    # filenm="CSL_D3_Lx"*string(Lx)*"_Ly"*string(Ly)*".jld2";
    # jldsave(filenm;psi);

    psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi_PEPS,psi_PEPS,Lx,Ly);


    multiplet_tol=1e-5;
    chi=100;

    global mpo_mps_trun_method, left_right_env_method;
    mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
    left_right_env_method="trun";#"exact","trun"

    global n_mps_sweep
    n_mps_sweep=5;


    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_old(psi_PEPS,psi_double)





    # jldsave("E_set_Lx"*string(Lx)*"_Ly"*string(Ly)*"_D_"*string(D)*".jld2"; E_set);

