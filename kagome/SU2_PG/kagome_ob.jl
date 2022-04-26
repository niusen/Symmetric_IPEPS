using LinearAlgebra
using TensorKit
using JSON
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model.jl")


D=8
chi=30
tol=1e-6

J1=1;
J2=0;
J3=0;
Jchi=0;
Jtrip=0;

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);



A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

#load elementary tensor coefficients from json file
function read_string(string)
    nums=ones(length(string))
    for cn=1:length(string)
        nums[cn]=parse.(Float64, split(string[cn])[2])
    end
    return nums
end
json_dict = Dict()
open("LS_D_"*string(D)*"_chi_40.json", "r") do f
    global json_dict
    dicttxt = read(f,String)  # file information to string
    json_dict=JSON.parse(dicttxt)  # parse and transform data
end

Bond_irrep=json_dict["Bond_irrep"]
Triangle_A1_coe=read_string(json_dict["coes"]["Triangle_A1_coe"]["entries"]);
Triangle_A2_coe=read_string(json_dict["coes"]["Triangle_A2_coe"]["entries"]);
if Bond_irrep=="A"
    Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);
elseif Bond_irrep=="B"
    Bond_B_coe=read_string(json_dict["coes"]["Bond_B_coe"]["entries"]);
elseif Bond_irrep=="A+iB"
    Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);
    Bond_B_coe=read_string(json_dict["coes"]["Bond_B_coe"]["entries"]);
end

global triangle_tensor=A1_set[1]*0;
global bond_tensor=A_set[1]*0;
if Bond_irrep=="A"
    global bond_tensor=A_set[1]*0;
    for ct=1:length(Bond_A_coe)
        global bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
    end
elseif Bond_irrep=="B"
    global bond_tensor=B_set[1]*0;
    for ct=1:length(Bond_B_coe)
        global bond_tensor=bond_tensor+B_set[ct]*Bond_B_coe[ct];
    end
elseif Bond_irrep=="A+iB"
    global bond_tensor=A_set[1]*0;
    for ct=1:length(Bond_A_coe)
        global bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
    end
    for ct=1:length(Bond_B_coe)
        global bond_tensor=bond_tensor+B_set[ct]*Bond_B_coe[ct];
    end
end

for ct=1:length(Triangle_A1_coe)
    global triangle_tensor=triangle_tensor+A1_set[ct]*Triangle_A1_coe[ct];
end
for ct=1:length(Triangle_A2_coe)
    global triangle_tensor=triangle_tensor+im*A2_set[ct]*Triangle_A2_coe[ct];
end

#TensorKit.usebraidcache_nonabelian[] = false
A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

#load elementary tensor coefficients from json file
function read_string(string)
    nums=ones(length(string));
    for cn=1:length(string)
        nums[cn]=parse.(Float64, split(string[cn])[2]);
    end
    return nums
end
json_dict = Dict()
open("LS_D_"*string(D)*"_chi_40.json", "r") do f
    global json_dict
    dicttxt = read(f,String)  # file information to string
    json_dict=JSON.parse(dicttxt)  # parse and transform data
end

Bond_irrep=json_dict["Bond_irrep"]
Triangle_A1_coe=read_string(json_dict["coes"]["Triangle_A1_coe"]["entries"]);
Triangle_A2_coe=read_string(json_dict["coes"]["Triangle_A2_coe"]["entries"]);
if Bond_irrep=="A"
    Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);
elseif Bond_irrep=="B"
    Bond_B_coe=read_string(json_dict["coes"]["Bond_B_coe"]["entries"]);
elseif Bond_irrep=="A+iB"
    Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);
    Bond_B_coe=read_string(json_dict["coes"]["Bond_B_coe"]["entries"]);
end



global triangle_tensor=A1_set[1]*0;
global bond_tensor=A_set[1]*0;
if Bond_irrep=="A"
    global bond_tensor=A_set[1]*0;
    for ct=1:length(Bond_A_coe)
        global bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
    end
elseif Bond_irrep=="B"
    global bond_tensor=B_set[1]*0;
    for ct=1:length(Bond_B_coe)
        global bond_tensor=bond_tensor+B_set[ct]*Bond_B_coe[ct];
    end
elseif Bond_irrep=="A+iB"
    global bond_tensor=A_set[1]*0;
    for ct=1:length(Bond_A_coe)
        global bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
    end
    for ct=1:length(Bond_B_coe)
        global bond_tensor=bond_tensor+B_set[ct]*Bond_B_coe[ct];
    end
end

for ct=1:length(Triangle_A1_coe)
    global triangle_tensor=triangle_tensor+A1_set[ct]*Triangle_A1_coe[ct];
end
for ct=1:length(Triangle_A2_coe)
    global triangle_tensor=triangle_tensor+im*A2_set[ct]*Triangle_A2_coe[ct];
end

PEPS_tensor=bond_tensor;
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];


init=Dict([("CTM", []), ("init_type", "PBC")]);
conv_check="singular_value";
@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(A_fused,chi,conv_check,tol,init);

@time E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, CTM, "E_triangle");
@time E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, CTM, "E_bond");




