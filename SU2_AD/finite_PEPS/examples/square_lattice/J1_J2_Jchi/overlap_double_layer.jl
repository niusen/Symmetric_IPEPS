using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\mps_methods.jl")
include("..\\..\\..\\environment\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\truncations.jl")

"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=4;
Ly=4;


data=load("CSL_D3.jld2");
A=data["A"];

psi_0_0=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
for cx=1:Lx
    for cy=1:Ly
        psi_0_0[cx,cy]=A;
    end
end

psi_0_pi=deepcopy(psi_0_0);
psi_pi_0=deepcopy(psi_0_0);
psi_pi_pi=deepcopy(psi_0_0);

gate_up=parity_gate(A,4);
for cx=1:Lx
    @tensor psi_0_pi[cx,Ly][:]:=psi_0_pi[cx,Ly][-1,-2,-3,1,-5]*gate_up[-4,1];
    @tensor psi_pi_pi[cx,Ly][:]:=psi_pi_pi[cx,Ly][-1,-2,-3,1,-5]*gate_up[-4,1];
end

gate_left=parity_gate(A,1);
for cy=1:Ly
    @tensor psi_pi_0[1,cy][:]:=psi_pi_0[1,cy][1,-2,-3,-4,-5]*gate_left[-1,1];
    @tensor psi_pi_pi[1,cy][:]:=psi_pi_pi[1,cy][1,-2,-3,-4,-5]*gate_left[-1,1];
end


psi_0_0=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_0_0));
psi_0_pi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_0_pi));
psi_pi_0=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_pi_0));
psi_pi_pi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_pi_pi));


psi_0_0_double=construct_double_layer(psi_0_0,psi_0_0);


mps_bot=psi_0_0_double[1:Lx,1];
norm_1D(mps_bot,mps_bot)
mps_con=right_canonical(mps_bot);
norm_1D(mps_con,mps_con)
#norm(psi_double)




multiplet_tol=1e-5;


chi=120;
Norm,trun_history=norm_2D_simple(psi_0_0_double,chi,multiplet_tol);
println(Norm)
println(trun_history)

