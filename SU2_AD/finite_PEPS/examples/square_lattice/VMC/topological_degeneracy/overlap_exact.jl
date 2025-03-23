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

include("../../../../setting/tuple_methods.jl")
include("../../../../symmetry/parity_funs.jl")
include("../../../../environment/AD/convert_boundary_condition.jl")
# include("../../../../environment/AD/mps_methods.jl")
# include("../../../../environment/AD/mps_methods_new.jl")
# include("../../../../environment/AD/peps_double_layer_methods.jl")
# include("../../../../environment/AD/peps_double_layer_methods_new.jl")
# include("../../../../environment/AD/truncations.jl")
# include("../../../../environment/AD/svd_AD_lib.jl")
# include("../../../../environment/simple_update/gauge_fix_spin.jl")
# include("../../../../environment/simple_update/simple_update_lib.jl")

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






function exact_contract(psi_)


    @tensor mps_row1[:]:=psi_[1,4][4,-5,1,-1,-9]*psi_[2,4][1,-6,2,-2,-10]*psi_[3,4][2,-7,3,-3,-11]*psi_[4,4][3,-8,4,-4,-12];
    U=unitary(fuse(space(psi_[1,1],5)*space(psi_[1,1],5)*space(psi_[1,1],5)*space(psi_[1,1],5)),  space(psi_[1,1],5)*space(psi_[1,1],5)*space(psi_[1,1],5)*space(psi_[1,1],5));
    @tensor mps_row1[:]:=mps_row1[-1,-2,-3,-4,-5,-6,-7,-8,1,2,3,4]*U[-9,1,2,3,4];


    @tensor mps_row2[:]:=psi_[1,3][4,-5,1,-1,-9]*psi_[2,3][1,-6,2,-2,-10]*psi_[3,3][2,-7,3,-3,-11]*psi_[4,3][3,-8,4,-4,-12];
    @tensor mps_row2[:]:=mps_row2[-1,-2,-3,-4,-5,-6,-7,-8,1,2,3,4]*U[-9,1,2,3,4];

    @tensor mps_row3[:]:=psi_[1,2][4,-5,1,-1,-9]*psi_[2,2][1,-6,2,-2,-10]*psi_[3,2][2,-7,3,-3,-11]*psi_[4,2][3,-8,4,-4,-12];
    @tensor mps_row3[:]:=mps_row3[-1,-2,-3,-4,-5,-6,-7,-8,1,2,3,4]*U[-9,1,2,3,4];

    @tensor mps_row4[:]:=psi_[1,1][4,-5,1,-1,-9]*psi_[2,1][1,-6,2,-2,-10]*psi_[3,1][2,-7,3,-3,-11]*psi_[4,1][3,-8,4,-4,-12];
    @tensor mps_row4[:]:=mps_row4[-1,-2,-3,-4,-5,-6,-7,-8,1,2,3,4]*U[-9,1,2,3,4];

    @tensor mps_12[:]:=mps_row1[-1,-2,-3,-4,1,2,3,4,-9]*mps_row2[1,2,3,4,-5,-6,-7,-8,-10];
    @tensor mps_34[:]:=mps_row3[-1,-2,-3,-4,1,2,3,4,-9]*mps_row4[1,2,3,4,-5,-6,-7,-8,-10];

    @tensor psi_total[:]:=mps_12[1,2,3,4,5,6,7,8,-1,-2]*mps_34[5,6,7,8,1,2,3,4,-3,-4];

    return psi_total
end


psi_0_0_total=exact_contract(psi_0_0);
psi_0_pi_total=exact_contract(psi_0_pi);
psi_pi_0_total=exact_contract(psi_pi_0);
psi_pi_pi_total=exact_contract(psi_pi_pi);

psis=(psi_0_0_total,psi_0_pi_total,psi_pi_0_total,psi_pi_pi_total,);

ov=zeros(4,4)*im;
for cx=1:4
    for cy=1:4
        ov[cx,cy]=dot(psis[cx],psis[cy]);
    end
end


for cx=1:4
    coe=sqrt(ov[cx,cx]);
    ov[cx,:]=ov[cx,:]/coe;
    ov[:,cx]=ov[:,cx]/coe;
end


matwrite("exact_ov.mat", Dict(
  "ov"=>ov
); compress = false)     


