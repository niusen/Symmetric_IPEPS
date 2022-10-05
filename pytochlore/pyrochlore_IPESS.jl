using TensorKit

#load elementary tensor coefficients from json file
function read_string(string)
    nums=ones(length(string))
    for cn=1:length(string)
        nums[cn]=parse.(Float64, split(string[cn])[2])
    end
    return nums
end
function read_json_state(filenm)
    json_dict = Dict()
    open(filenm, "r") do f
        json_dict
        dicttxt = read(f,String)  # file information to string
        json_dict=JSON.parse(dicttxt)  # parse and transform data
    end
    return json_dict
end


function construct_su2_PG_IPESS(json_dict,A_set,E_set, S_label, Sz_label, virtual_particle, Va, Vb)
    Bond_irrep=json_dict["Bond_irrep"]

    Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);

    Tetrahedral_irrep=json_dict["Tetrahedral_irrep"]
    Tetrahedral_E_coe=read_string(json_dict["coes"]["Tetrahedral_E_coe"]["entries"]);




    #combine tensors with coefficients
    bond_tensor=A_set[1]*0;
    for ct=1:length(Bond_A_coe)
        bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
    end

    Tetrahedral_tensor=E_set[1]*0;
    for ct=1:length(Tetrahedral_E_coe)
        Tetrahedral_tensor=Tetrahedral_tensor+E_set[ct]*Tetrahedral_E_coe[ct];
    end

    return bond_tensor,Tetrahedral_tensor
end




function get_tensor_coes(json_dict)
    Bond_irrep=json_dict["Bond_irrep"]
    Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);

    Tetrahedral_irrep=json_dict["Tetrahedral_irrep"]
    Tetrahedral_E_coe=read_string(json_dict["coes"]["Tetrahedral_E_coe"]["entries"]);

    return Bond_irrep, Tetrahedral_irrep, Bond_A_coe, Tetrahedral_E_coe
end