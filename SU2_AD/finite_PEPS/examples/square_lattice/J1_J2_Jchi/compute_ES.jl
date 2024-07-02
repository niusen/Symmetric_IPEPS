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

include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")

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
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\optimization\\PEPS_methods.jl")

include("..\\..\\..\\environment\\full_update\\full_update_lib.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\triangle_terms.jl")
include("..\\..\\..\\environment\\simple_update\\simple_update_lib.jl")

Random.seed!(888)

filenm="optim_4x4_D_3_chi_100_13.89035.jld2";

global use_AD;
use_AD=false;

global chi
chi=100;

D_max=3;


J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
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

multiplet_tol=1e-5;
global chi,multiplet_tol

global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;



# data=load(filenm);
# psi=data["psi"];
# psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

init_noise=0;
psi=initial_SU2_state(filenm,init_noise,true);
psi_double=construct_double_layer(psi, nothing);

global Lx,Ly
Lx,Ly=size(psi)

#################################################
global chi, multiplet_tol

#     global chi, multiplet_tol

global mpo_mps_trun_method, left_right_env_method;
if mpo_mps_trun_method=="canonical"
    mpo_mps_fun=truncate_mpo_mps;
elseif mpo_mps_trun_method=="exact"
        mpo_mps_fun=truncate_mpo_mps_exact;
elseif mpo_mps_trun_method=="simple_middle"
    mpo_mps_fun=simple_truncate_to_moddle;
end




########################################
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
    mps_bot,trun_errs,_=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
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
    mps_top,trun_errs,_=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
    mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
    trun_history=vcat(trun_history,trun_errs);
end


###################################
mps_top=mps_top_set[Int(Ly/2+1)];
mps_bot=mps_bot_set[Int(Ly/2)];
mps_top=collect(mps_top);
mps_bot=collect(mps_bot);
for cx=1:Lx
    cy=Int(Ly/2);
    # A=psi[cx,cy];
    # U_U=unitary(space(A, 4) ⊗ space(A, 4)', fuse(space(A, 4)' ⊗ space(A, 4)));
    _,U_L,U_D,U_R,U_U=construct_double_layer_pos(psi,psi,cx,cy);
    println(space(U_U))
    if cx==1
        T=mps_bot[cx];
        @tensor T[:]:=T[-1,1]*U_U'[1,-2,-3];
        mps_bot[cx]=T;

        T=mps_top[cx];
        @tensor T[:]:=T[-1,1]*U_U[-2,-3,1];
        mps_top[cx]=T;        
    elseif 1<cx<Lx
        T=mps_bot[cx];
        @tensor T[:]:=T[-1,-2,1]*U_U'[1,-3,-4];
        mps_bot[cx]=T;

        T=mps_top[cx];
        @tensor T[:]:=T[-1,-2,1]*U_U[-3,-4,1];
        mps_top[cx]=T;        
    elseif cx==Lx
        T=mps_bot[cx];
        @tensor T[:]:=T[-1,1]*U_U'[1,-2,-3];
        mps_bot[cx]=T;

        T=mps_top[cx];
        @tensor T[:]:=T[-1,1]*U_U[-2,-3,1];
        mps_top[cx]=T;        
    end
end
###################################
#direct contraction
@tensor Tup[:]:=mps_top[1][1,-1,-5]*mps_top[2][1,2,-2,-6]*mps_top[3][2,3,-3,-7]*mps_top[4][3,-4,-8];
@tensor Tdn[:]:=mps_bot[1][1,-1,-5]*mps_bot[2][1,2,-2,-6]*mps_bot[3][2,3,-3,-7]*mps_bot[4][3,-4,-8];

@tensor H[:]:=Tup[-1,-2,-3,-4,1,2,3,4]*Tdn[-5,-6,-7,-8,1,2,3,4];
H=permute(H,(1,2,3,4,),(5,6,7,8,));

eu,ev=eigen(H);

function get_ES(v0)
    eu0,ev0=eigen(v0);

    Spin_set=Vector{Int64}(undef,0);
    eu_set=Vector{Int64}(undef,0);
    for cc=1:length(eu0.data.values)
      spa=eu0.data.keys[cc];

      Spin=spa.j;
      mm=diag(eu0.data.values[cc]);
      Spin_set=vcat(Spin_set,Spin*ones(length(mm)));
      eu_set=vcat(eu_set,mm);

    end

    eu_set=eu_set/sum(abs.(eu_set));
    pos=findall(x-> x.>1e-6, abs.(eu_set));
    Spin_set=Spin_set[pos];
    eu_set=eu_set[pos];

    order=sortperm(abs.(eu_set));
    Spin_set=Spin_set[order];
    eu_set=eu_set[order];
    return Spin_set,eu_set
  end

  Spin_set,eu_set=get_ES(eu);






