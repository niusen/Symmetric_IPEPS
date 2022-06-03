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




function construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb)
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


    triangle_tensor=A1_set[1]*0;
    bond_tensor=A_set[1]*0;
    if Bond_irrep=="A"
        bond_tensor=A_set[1]*0;
        for ct=1:length(Bond_A_coe)
            bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
        end
    elseif Bond_irrep=="B"
        bond_tensor=B_set[1]*0;
        for ct=1:length(Bond_B_coe)
            bond_tensor=bond_tensor+B_set[ct]*Bond_B_coe[ct];
        end
    elseif Bond_irrep=="A+iB"
        bond_tensor=A_set[1]*0;
        for ct=1:length(Bond_A_coe)
            bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
        end
        for ct=1:length(Bond_B_coe)
            bond_tensor=bond_tensor+B_set[ct]*Bond_B_coe[ct];
        end
    end

    for ct=1:length(Triangle_A1_coe)
        triangle_tensor=triangle_tensor+A1_set[ct]*Triangle_A1_coe[ct];
    end
    for ct=1:length(Triangle_A2_coe)
        triangle_tensor=triangle_tensor+im*A2_set[ct]*Triangle_A2_coe[ct];
    end
    return bond_tensor,triangle_tensor
end