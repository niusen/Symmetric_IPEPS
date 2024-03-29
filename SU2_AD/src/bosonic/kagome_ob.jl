using LinearAlgebra
using TensorKit
using JSON
using Zygote:@ignore_derivatives
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model.jl")
include("kagome_IPESS.jl")

D=8
chi=20
tol=1e-6

J1=0.80902;
J2=0;
J3=0;
Jchi=0;
Jtrip=0.5878;

# J1=1;
# J2=0;
# J3=0;
# Jchi=0;
# Jtrip=1;

CTM_ite_nums=150;
CTM_trun_tol=1e-14;


parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);

A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

filenm="LS_D_"*string(D)*"_chi_40.json"
json_dict=read_json_state(filenm);

bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

PEPS_tensor=bond_tensor;
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
A_unfused=PEPS_tensor;

U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

# A_fused=permute(A_fused,(2,3,4,1,5,),())#to compare with python
# A_unfused=permute(A_unfused,(2,3,4,1,5,6,7,),())#to compare with python




init=Dict([("CTM", []), ("init_type", "PBC")]);
conv_check="singular_value";
@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(A_fused,chi,conv_check,tol,init,CTM_ite_nums,CTM_trun_tol);

@time E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
@time E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_bond");

display((E_up+E_down)/3)

display(space(CTM["Cset"][1]))
display(space(CTM["Cset"][2]))
display(space(CTM["Cset"][3]))
display(space(CTM["Cset"][4]))



# matwrite("Corners_julia"*"_D"*string(D)*"_chi"*string(chi)*".mat", Dict(
#     "C1" => svdvals(convert(Array, CTM["Cset"][1])),
#     "C2" => svdvals(convert(Array, CTM["Cset"][2])),
#     "C3" => svdvals(convert(Array, CTM["Cset"][3])),
#     "C4" => svdvals(convert(Array, CTM["Cset"][4]))
# ); compress = false)
