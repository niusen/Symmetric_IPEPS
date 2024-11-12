using Distributed
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
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell_iPESS_speed.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate_iPESS.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_FullUpdate_iPESS.jl")
include("..\\..\\src\\fermionic\\simple_update\\Full_Update_lib.jl")
include("..\\..\\src\\fermionic\\fermion_ob_iPESS.jl")



###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################

# Random.seed!(888);
#V=Rep[SU₂](0=>30, 1/2=>15,1=>8,3/2=>8,2=>8);
# V=Rep[ℤ₂](0=>25,1=>25);
# V=Rep[U₁ × SU₂]((0, 0)=>10, (2, 0)=>10, (1, 1/2)=>5,(2, 1)=>5,(3, 1/2)=>5);
V=Rep[U₁](-1=>10, -2=>20, 1=>10,0=>10,2=>10);
M=TensorMap(randn,V*V,V*V);
# M=TensorMap(randn,V,V);



chi=Int(round(dim(V)/2));

uM1,sM1,vM1 = my_tsvd(M; trunc=truncdim(chi));



N_blocks=length(M.data.values);
uu_set=Vector{Array}(undef,N_blocks);
ss_set=Vector{Array}(undef,N_blocks);
vv_set=Vector{Array}(undef,N_blocks);
for ccc =1:N_blocks
    uu,ss,vv = svd(M.data.values[ccc]);
    vv=vv';
    uu_set[ccc]=uu;
    ss_set[ccc]=ss;
    vv_set[ccc]=vv;
end
# println(ss_set)
uM2,sM2,vM2 = truncate_block_svd(uu_set,ss_set,vv_set,M,chi);


@assert norm(uM1*sM1*vM1-uM2*sM2*vM2)/norm(M)<1e-13;
@show norm(uM1*sM1*vM1-uM2*sM2*vM2)/norm(M)