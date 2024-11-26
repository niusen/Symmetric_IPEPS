using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\environment\\MC\\truncations.jl")
include("..\\..\\..\\environment\\MC\\mps_methods.jl")
include("..\\..\\..\\environment\\MC\\mps_methods_new.jl")
include("..\\..\\..\\environment\\MC\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\MC\\density_matrix.jl")
include("..\\..\\..\\environment\\MC\\density_matrix_new.jl")
include("..\\..\\..\\environment\\MC\\sampling_eliminate_physical_leg.jl")

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")

include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")

include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\ansatz\\square_lattice\\square_lattice.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\environment\\simple_update\\gauge_fix_spin.jl")
include("..\\..\\..\\environment\\simple_update\\simple_update_lib.jl")


Random.seed!(888);
global use_AD;
use_AD=false;



D=3;

#Hamiltonian
H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

#######################################
global use_AD;
use_AD=false;

global chi,multiplet_tol
chi=100;
multiplet_tol=1e-5;


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
#######################################
global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=0;
##############################################

Lx=4;
Ly=4;

data=load("CSL_D3_SU2.jld2");
A=data["A"];

P=zeros(1,3);P[1,1]=1;
P_L=TensorMap(P,Rep[SU₂](0=>1),space(A,1));
P_D=TensorMap(P,Rep[SU₂](0=>1),space(A,2));

psi=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
for cx=2:Lx-1
    for cy=2:Ly-1
        psi[cx,cy]=A;
    end
end

cx=1;
for cy=2:Ly-1
    @tensor T[:]:=A[1,-2,-3,-4,-5]*P_L[-1,1];
    psi[cx,cy]=T;
end

cx=Lx;
for cy=2:Ly-1
    @tensor T[:]:=A[-1,-2,1,-4,-5]*P_L'[1,-3];
    psi[cx,cy]=T;
end

cy=1;
for cx=2:Lx-1
    @tensor T[:]:=A[-1,1,-3,-4,-5]*P_D[-2,1];
    psi[cx,cy]=T;
end

cy=Ly;
for cx=2:Lx-1
    @tensor T[:]:=A[-1,-2,-3,1,-5]*P_D'[1,-4];
    psi[cx,cy]=T;
end

cx=1;
cy=1;
@tensor T[:]:=A[1,2,-3,-4,-5]*P_L[-1,1]*P_D[-2,2];
psi[cx,cy]=T;

cx=Lx;
cy=1;
@tensor T[:]:=A[-1,2,1,-4,-5]*P_L'[1,-3]*P_D[-2,2];
psi[cx,cy]=T;

cx=1;
cy=Ly;
@tensor T[:]:=A[1,-2,-3,2,-5]*P_L[-1,1]*P_D'[2,-4];
psi[cx,cy]=T;

cx=Lx;
cy=Ly;
@tensor T[:]:=A[-1,-2,1,2,-5]*P_L'[1,-3]*P_D'[2,-4];
psi[cx,cy]=T;





psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

filenm="CSL_D3_Lx"*string(Lx)*"_Ly"*string(Ly)*".jld2";
# jldsave(filenm;psi);

psi_double=construct_double_layer(psi,psi);








global mpo_mps_trun_method;
mpo_mps_trun_method="simple_middle";#"simple_middle",""canonical""

psi_network=Matrix{TensorMap}(undef,Lx,Ly)

for cx=1:Lx
    for cy=1:Ly
        Treal=TensorMap(randn,codomain(psi_double[cx,cy]),domain(psi_double[cx,cy]));
        Timag=TensorMap(randn,codomain(psi_double[cx,cy]),domain(psi_double[cx,cy]));
        T=Treal+Timag*im;
        T=T/norm(T);
        psi_network[cx,cy]=T;
    end
end




@time Norm11,trunerr11=norm_2D_simple(psi_network,chi,multiplet_tol)
@show sum(abs.(trunerr11))
@time Norm12,trunerr12=overlap(psi_network);#n_mps_sweep is set to zero in this function
@show sum(abs.(trunerr12))




psi1=add_trivial_physical_leg(psi_network)

psi1=disk_to_torus(psi1);
psi1_with_coe=finite_PEPS_with_coe(psi1,0);
psi1_with_coe,_=PEPS_gauge_fix_simple(psi1_with_coe,100);
psi1=remove_trivial_boundary_leg(psi1_with_coe.Tset);

psi1=remove_trivial_physical_leg(psi1);


@time Norm21,trunerr21=norm_2D_simple(psi1,chi,multiplet_tol)
@time Norm22,trunerr22=overlap(psi1);
@show sum(abs.(trunerr21))
@show sum(abs.(trunerr22))



