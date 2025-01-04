using LinearAlgebra:I,diagm,diag
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
include("..\\..\\..\\environment\\MC\\contract_torus.jl")
include("..\\..\\..\\environment\\MC\\build_degenerate_states.jl")

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


include("..\\..\\..\\environment\\MC\\sampling.jl")
include("..\\..\\..\\environment\\MC\\sampling_eliminate_physical_leg.jl")

"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=8;
Ly=8;

#note: when copy a variable, use deepcopy()

#######################################
#set parameters
global use_AD;
use_AD=false;

global chi,multiplet_tol
chi=40;
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

data=load("CSL_D3_U1.jld2");
A=data["A"];


Vv=U₁Space(0=>1,1/2=>1,-1/2=>1);
Vp=U₁Space(1/2=>1,-1/2=>1);
# Vv=ℤ₂Space(0=>1,1=>2);
# Vp=ℤ₂Space(1=>2);
A=TensorMap(convert(Array,A),Vv*Vv,Vv*Vv*Vp);
A=permute(A,(1,2,3,4,5,));


#psi=generate_obc_from_iPEPS(A,Lx,Ly);
psi=Matrix{TensorMap}(undef,Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        psi[cx,cy]=A;
    end
end

psi_00,psi_0pi,psi_pi0,psi_pipi =construct_4_states(psi,Vv);#four states
#################################

#initial spin config, total sz=0
config=zeros(Int8,Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        config[cx,cy]=(-1)^(cx+cy);
    end
end


#apply projector to obtain sample
psi_sample=apply_sampling_projector(psi,config);
psi_sample=shift_pleg(psi_sample);


global projector_method
projector_method="2";#"1" or "2"

#do contraction
@time Norm,trun_err=contract_whole_torus(psi_sample,chi);
@show [Norm,sum(abs.(trun_err))]
# ##############################

