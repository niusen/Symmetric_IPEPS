using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
using LineSearches,OptimKit
cd(@__DIR__)

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\state\\FinitePEPS.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
include("..\\..\\..\\environment\\extend_bond\\extend_bond.jl")
include("..\\..\\..\\environment\\extend_bond\\environment_2site.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator_dense.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\optimization\\PEPS_methods.jl")
include("..\\..\\..\\optimization\\LineSearches\\My_Backtracking.jl")


Random.seed!(888)
global use_AD;
use_AD=false;

global chi
chi=100;

filenm="WYLiu_D2.jld2";


J1=1;
J2=0;
Jchi=0;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
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




"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""





data=load(filenm);
psi=data["psi"];
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

global Lx,Ly
Lx=size(psi,1);
Ly=size(psi,2);

psi_double=construct_double_layer(psi,psi);


global E_history
E_history=[10000];





multiplet_tol=1e-5;



global chi,multiplet_tol



global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;


E_opt=real(cost_fun_global(psi));
println(E_opt)



E_total,E_set=energy_disk_(psi)



#########################################
psi_double=construct_double_layer(psi,psi);


multiplet_tol=1e-5;
chi=60;
############################################
mpo_mps_fun=simple_truncate_to_moddle;


    #construct top and bot environment

    trun_history=[];
    mps_bot_set=initial_tuple(Ly);
    mps_top_set=initial_tuple(Ly);

    mps_bot=(psi_double[:,1]...,);
    mps_bot_set=vector_update(mps_bot_set,mps_bot,1);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:Ly-2
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
        end
        return mps_top
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),Ly);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:3
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
        mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
        trun_history=vcat(trun_history,trun_errs);
    end

    #global trun_history
    # println(trun_history)
    ########################################

    #construct left anf right environment
    
    VL_set_set=initial_tuple(Ly);
    VR_set_set=initial_tuple(Ly);

    cy=1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+2];
    mpo_top=(psi_double[:,cy+1]...,);
    mps_bot=mps_bot_set[cy];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);

    for cy=2:Ly-2
        VL_set=initial_tuple(Lx);
        VR_set=initial_tuple(Lx);
        mps_top=mps_top_set[cy+2];
        mpo_top=(psi_double[:,cy+1]...,);
        mpo_bot=(psi_double[:,cy]...,);
        mps_bot=mps_bot_set[cy-1];
        if left_right_env_method=="exact"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            VL_set=vector_update(VL_set,vl,1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl[1,3,5,7]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mpo_bot[cx][5,6,-3,4]*mps_bot[cx][7,-4,6];
                VL_set=vector_update(VL_set,vl,cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            VR_set=vector_update(VR_set,vr,Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr[1,3,5,7]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mpo_bot[cx][-3,6,5,4]*mps_bot[cx][-4,7,6];
                VR_set=vector_update(VR_set,vr,cx);
            end
        elseif left_right_env_method=="trun"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            vl_up,vl_dn=split_vl_or_vr(vl);
            VL_set=vector_update(VL_set,(vl_up,vl_dn,),1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl_up[4,6,7]*vl_dn[7,2,1]*mps_top[cx][4,-1,5]*mpo_top[cx][6,8,-2,5]*mpo_bot[cx][2,3,-3,8]*mps_bot[cx][1,-4,3];
                vl_up,vl_dn=split_vl_or_vr(vl);
                VL_set=vector_update(VL_set,(vl_up,vl_dn,),cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            vr_up,vr_dn=split_vl_or_vr(vr);
            VR_set=vector_update(VR_set,(vr_up,vr_dn,),Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr_up[1,3,8]*vr_dn[8,5,4]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,7,3,2]*mpo_bot[cx][-3,6,5,7]*mps_bot[cx][-4,4,6];
                vr_up,vr_dn=split_vl_or_vr(vr);
                VR_set=vector_update(VR_set,(vr_up,vr_dn,),cx);
            end
        end
        VL_set_set=vector_update(VL_set_set,VL_set,cy);
        VR_set_set=vector_update(VR_set_set,VR_set,cy);
    end

    cy=Ly-1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+1];
    mpo_bot=(psi_double[:,cy]...,);
    mps_bot=mps_bot_set[cy-1];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_bot[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_bot[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_bot[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_bot[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);





##############################################

function normalize_rho(rho,U_s_s)
    @tensor rho[:]:=rho[1,2,3,4]*U_s_s[-1,-5,1]*U_s_s[-2,-6,2]*U_s_s[-3,-7,3]*U_s_s[-4,-8,4];
    rho=permute(rho,(1,2,3,4,),(5,6,7,8,));
    Norm=@tensor rho[1,2,3,4,1,2,3,4];
    rho=rho/Norm;
    return rho
end


x_range=[2,3];
y_range=[2,3];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
rho_bulk=normalize_rho(rho,U_s_s);

x_range=[1,2];
y_range=[2,3];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
rho_left=normalize_rho(rho,U_s_s);

x_range=[Lx-1,Lx];
y_range=[2,3];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
rho_right=normalize_rho(rho,U_s_s);

x_range=[2,3];
y_range=[Ly-1,Ly];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
rho_top=normalize_rho(rho,U_s_s);

x_range=[2,3];
y_range=[1,2];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
rho_bot=normalize_rho(rho,U_s_s);

x_range=[1,2];
y_range=[Ly-1,Ly];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
rho_left_top=normalize_rho(rho,U_s_s);

x_range=[Lx-1,Lx];
y_range=[Ly-1,Ly];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
rho_right_top=normalize_rho(rho,U_s_s);

x_range=[1,2];
y_range=[1,2];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
rho_left_bot=normalize_rho(rho,U_s_s);

x_range=[Lx-1,Lx];
y_range=[1,2];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
rho_right_bot=normalize_rho(rho,U_s_s);



println(norm(rho_bulk-permute(rho_bulk,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_top-permute(rho_right,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_right-permute(rho_bot,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_bot-permute(rho_left,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_left-permute(rho_top,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_left_top-permute(rho_right_top,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_right_top-permute(rho_right_bot,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_right_bot-permute(rho_left_bot,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_left_bot-permute(rho_left_top,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))



H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");



E_x=zeros(3,4)*im*1.0;

E_x[1,1]=@tensor rho_left_bot[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[1,2]=@tensor rho_left[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[1,3]=@tensor rho_left_top[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[1,4]=@tensor rho_left_top[1,2,5,6,3,4,5,6]*H_Heisenberg[1,2,3,4];

E_x[2,1]=@tensor rho_bot[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[2,2]=@tensor rho_bulk[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[2,3]=@tensor rho_top[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[2,4]=@tensor rho_top[1,2,5,6,3,4,5,6]*H_Heisenberg[1,2,3,4];

E_x[3,1]=@tensor rho_right_bot[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[3,2]=@tensor rho_right[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[3,3]=@tensor rho_right_top[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[3,4]=@tensor rho_right_top[1,2,5,6,3,4,5,6]*H_Heisenberg[1,2,3,4];


E_y=zeros(4,3)*im*1.0;

E_y[1,1]=@tensor rho_left_bot[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[2,1]=@tensor rho_bot[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[3,1]=@tensor rho_right_bot[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[4,1]=@tensor rho_right_bot[5,1,2,6,5,3,4,6]*H_Heisenberg[1,2,3,4];

E_y[1,2]=@tensor rho_left[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[2,2]=@tensor rho_bulk[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[3,2]=@tensor rho_right[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[4,2]=@tensor rho_right[5,1,2,6,5,3,4,6]*H_Heisenberg[1,2,3,4];

E_y[1,3]=@tensor rho_left_top[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[2,3]=@tensor rho_top[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[3,3]=@tensor rho_right_top[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[4,3]=@tensor rho_right_top[5,1,2,6,5,3,4,6]*H_Heisenberg[1,2,3,4];


E_right_top=zeros(3,3)*im*1.0;

E_right_top[1,1]=@tensor rho_left_bot[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[2,1]=@tensor rho_bot[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[3,1]=@tensor rho_right_bot[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];

E_right_top[1,2]=@tensor rho_left[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[2,2]=@tensor rho_bulk[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[3,2]=@tensor rho_right[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];

E_right_top[1,3]=@tensor rho_left_top[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[2,3]=@tensor rho_top[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[3,3]=@tensor rho_right_top[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];


E_right_bot=zeros(3,3)*im*1.0;

E_right_bot[1,1]=@tensor rho_left_bot[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[2,1]=@tensor rho_bot[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[3,1]=@tensor rho_right_bot[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];

E_right_bot[1,2]=@tensor rho_left[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[2,2]=@tensor rho_bulk[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[3,2]=@tensor rho_right[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];

E_right_bot[1,3]=@tensor rho_left_top[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[2,3]=@tensor rho_top[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[3,3]=@tensor rho_right_top[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];


E_chiral_123=zeros(3,3)*im*1.0;

E_chiral_123[1,1]=@tensor rho_left_bot[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[2,1]=@tensor rho_bot[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[3,1]=@tensor rho_right_bot[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];

E_chiral_123[1,2]=@tensor rho_left[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[2,2]=@tensor rho_bulk[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[3,2]=@tensor rho_right[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];

E_chiral_123[1,3]=@tensor rho_left_top[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[2,3]=@tensor rho_top[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[3,3]=@tensor rho_right_top[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];


E_chiral_234=zeros(3,3)*im*1.0;

E_chiral_234[1,1]=@tensor rho_left_bot[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[2,1]=@tensor rho_bot[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[3,1]=@tensor rho_right_bot[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];

E_chiral_234[1,2]=@tensor rho_left[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[2,2]=@tensor rho_bulk[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[3,2]=@tensor rho_right[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];

E_chiral_234[1,3]=@tensor rho_left_top[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[2,3]=@tensor rho_top[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[3,3]=@tensor rho_right_top[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];


E_chiral_341=zeros(3,3)*im*1.0;

E_chiral_341[1,1]=@tensor rho_left_bot[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[2,1]=@tensor rho_bot[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[3,1]=@tensor rho_right_bot[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];

E_chiral_341[1,2]=@tensor rho_left[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[2,2]=@tensor rho_bulk[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[3,2]=@tensor rho_right[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];

E_chiral_341[1,3]=@tensor rho_left_top[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[2,3]=@tensor rho_top[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[3,3]=@tensor rho_right_top[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];


E_chiral_412=zeros(3,3)*im*1.0;

E_chiral_412[1,1]=@tensor rho_left_bot[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[2,1]=@tensor rho_bot[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[3,1]=@tensor rho_right_bot[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];

E_chiral_412[1,2]=@tensor rho_left[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[2,2]=@tensor rho_bulk[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[3,2]=@tensor rho_right[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];

E_chiral_412[1,3]=@tensor rho_left_top[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[2,3]=@tensor rho_top[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[3,3]=@tensor rho_right_top[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];


E=J1*sum(E_x)+J1*sum(E_y)+J2*sum(E_right_bot)+J2*sum(E_right_top)+Jchi*sum(E_chiral_123+E_chiral_234+E_chiral_341+E_chiral_412)



