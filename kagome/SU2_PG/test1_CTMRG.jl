using LinearAlgebra
using TensorKit
using JSON
cd("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("load_tensor.jl")
include("CTM.jl")
#include("my_TensorKit_fun.jl")

D=3
chi=10
tol=1e-6


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
        global bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct]
    end
elseif Bond_irrep=="B"
    global bond_tensor=B_set[1]*0;
    for ct=1:length(Bond_B_coe)
        global bond_tensor=bond_tensor+B_set[ct]*Bond_B_coe[ct]
    end
elseif Bond_irrep=="A+iB"
    global bond_tensor=A_set[1]*0;
    for ct=1:length(Bond_A_coe)
        global bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct]
    end
    for ct=1:length(Bond_B_coe)
        global bond_tensor=bond_tensor+B_set[ct]*Bond_B_coe[ct]
    end
end

for ct=1:length(Triangle_A1_coe)
    global triangle_tensor=triangle_tensor+A1_set[ct]*Triangle_A1_coe[ct]
end
for ct=1:length(Triangle_A2_coe)
    global triangle_tensor=triangle_tensor+im*A2_set[ct]*Triangle_A2_coe[ct]
end

@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor PEPS_tensor[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

#PEPS_tensor_dense=convert(Array,PEPS_tensor)

#save initial CTM to compare with other codes
CTM=init_CTM(10,PEPS_tensor,"PBC");
matwrite("matfile.mat", Dict(
	"C1" => convert(Array,CTM["Cset"][1]),
	"C2" => convert(Array,CTM["Cset"][2]),
    "C3" => convert(Array,CTM["Cset"][3]),
    "C4" => convert(Array,CTM["Cset"][4]),
    "T1" => convert(Array,CTM["Tset"][1]),
    "T2" => convert(Array,CTM["Tset"][2]),
    "T3" => convert(Array,CTM["Tset"][3]),
    "T4" => convert(Array,CTM["Tset"][4])
); compress = false)

init=Dict([("CTM", []), ("init_type", "PBC")])
CTM=CTMRG(PEPS_tensor,chi,"singular_value",tol,init);

# ob_opts=Dict([("SiteNumber", 1), ("direction", "x")])
# rho=ob_CTMRG(CTM,PEPS_tensor,ob_opts)
# rho=permute(rho,(1,),(2,))
# #display(reshape(convert(Array,rho),8,8))

# # Heisenberg interaction
# Id=I(2);
# sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
# @tensor H[:]:=sx[-1,-4]*sx[-2,-5]*Id[-3,-6]+sy[-1,-4]*sy[-2,-5]*Id[-3,-6]+sz[-1,-4]*sz[-2,-5]*Id[-3,-6];
# rho=convert(Array,rho);
# @tensor E[:]:=rho[1,2,3,4,5,6]*H[4,5,6,1,2,3];
# E=E[1]*2;
# display("energy is: "*string(E))



