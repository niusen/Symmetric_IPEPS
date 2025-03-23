using LinearAlgebra:I,diagm,diag
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("../../../../../src/bosonic/iPEPS_ansatz.jl")
# include("../../../../../src/bosonic/line_search_lib.jl")
# include("../../../../../src/bosonic/line_search_lib_cell.jl")

include("../../../../setting/Settings.jl")
include("../../../../optimization/line_search_lib.jl")
# include("../../../../state/FinitePEPS.jl")
include("../../../../setting/tuple_methods.jl")
include("../../../../symmetry/parity_funs.jl")
include("../../../../environment/AD/convert_boundary_condition.jl")
include("../../../../environment/AD/mps_methods.jl")
include("../../../../environment/AD/mps_methods_new.jl")
include("../../../../environment/AD/peps_double_layer_methods.jl")
include("../../../../environment/AD/peps_double_layer_methods_new.jl")
include("../../../../environment/AD/truncations.jl")
include("../../../../environment/AD/svd_AD_lib.jl")
include("../../../../environment/simple_update/gauge_fix_spin.jl")
include("../../../../environment/simple_update/simple_update_lib.jl")

include("../../../../environment/MC/build_degenerate_states.jl")

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
use_canonical_form=true;

global use_canonical_form

###########################################
global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;
##############################################
Lx=4;
Ly=4;


data=load("CSL_D3_U1.jld2");
A=data["A"];
A_dense=convert(Array,A)
Vv=Rep[SU₂](0=>1, 1/2=>1);
Vp=Rep[SU₂](1/2=>1);
A=TensorMap(A_dense,Vv*Vv,Vv*Vv*Vp);
A=permute(A,(1,2,3,4,5,));

psi0=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
for cx=1:Lx
    for cy=1:Ly
        psi0[cx,cy]=A;
    end
end

Vv=space(psi0[1,1],1);
psi_0_0,psi_0_pi,psi_pi_0,psi_pi_pi=construct_4_states(psi0,Vv);#four states




psi_0_0=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_0_0));
psi_0_pi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_0_pi));
psi_pi_0=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_pi_0));
psi_pi_pi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_pi_pi));




psi_double=construct_double_layer(psi_0_0,psi_0_pi);

multiplet_tol=1e-5;


chi=81;
Norm,trun_history=norm_2D_simple(psi_double,chi,multiplet_tol);
println(Norm)
println(trun_history)

