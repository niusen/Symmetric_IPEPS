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

function has_odd(occus,virtual_particle)
    #counts the occupation number of half inter spins in A1 and A2 triangle tensors
    posit=inds=findall(x->x==0.5, virtual_particle.%1);
    if sum(occus[posit])==0
        return 0
    elseif sum(occus[posit])==2
        return 1
    else
        error("incorrect number of virtual half integer spins") 
    end
end


function nonchiral_projection(nonchiral,Triangle_A1_coe,Triangle_A2_coe,A1_set_occu,A2_set_occu,virtual_particle)
    A1_has_odd=Vector{Float64}(undef, length(A1_set_occu));
    A2_has_odd=Vector{Float64}(undef, length(A2_set_occu));
    for ct=1:length(A1_set_occu)
        A1_has_odd[ct]=has_odd(A1_set_occu[ct],virtual_particle);
    end
    for ct=1:length(A2_set_occu)
        A2_has_odd[ct]=has_odd(A2_set_occu[ct],virtual_particle);
    end

        
    #projection operation for nonchiral states. Options: nothing,"A1_even","A1_odd"
    if nonchiral=="No"
    elseif nonchiral=="A1_even"
        for ct=1:length(Triangle_A1_coe)
            Triangle_A1_coe[ct]=Triangle_A1_coe[ct]*(1-A1_has_odd[ct])
        end
        for ct=1:length(Triangle_A2_coe)
            Triangle_A2_coe[ct]=Triangle_A2_coe[ct]*A2_has_odd[ct]
        end
    elseif nonchiral=="A1_odd"
        for ct=1:length(Triangle_A1_coe)
            Triangle_A1_coe[ct]=Triangle_A1_coe[ct]*A1_has_odd[ct]
        end
        for ct=1:length(Triangle_A2_coe)
            Triangle_A2_coe[ct]=Triangle_A2_coe[ct]*(1-A2_has_odd[ct])
        end
    end
    return Triangle_A1_coe,Triangle_A2_coe, A1_has_odd, A2_has_odd
end

function construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb)
    Bond_irrep=json_dict["Bond_irrep"]
    nonchiral=json_dict["nonchiral"]

    if Bond_irrep=="A"
        Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);
        Bond_B_coe=[];
    elseif Bond_irrep=="B"
        Bond_A_coe=[];
        Bond_B_coe=read_string(json_dict["coes"]["Bond_B_coe"]["entries"]);
    elseif Bond_irrep=="A+iB"
        Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);
        Bond_B_coe=read_string(json_dict["coes"]["Bond_B_coe"]["entries"]);
    end

    Triangle_irrep=json_dict["Triangle_irrep"]
    if Triangle_irrep=="A1"
        Triangle_A1_coe=read_string(json_dict["coes"]["Triangle_A1_coe"]["entries"]);
        Triangle_A2_coe=[];
    elseif Triangle_irrep=="A2"
        Triangle_A1_coe=[];
        Triangle_A2_coe=read_string(json_dict["coes"]["Triangle_A2_coe"]["entries"]);
    elseif Triangle_irrep=="A1+iA2"
        Triangle_A1_coe=read_string(json_dict["coes"]["Triangle_A1_coe"]["entries"]);
        Triangle_A2_coe=read_string(json_dict["coes"]["Triangle_A2_coe"]["entries"]);

        
    end



    @assert length(Bond_A_coe)==length(A_set)


    #combine tensors with coefficients
    bond_tensor=A_set[1]*0;
    if Bond_irrep=="A"
        bond_tensor=A_set[1]*0;
        for ct=1:length(Bond_A_coe)
            bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
        end
    elseif Bond_irrep=="B"
        bond_tensor=B_set[1]*0;
        for ct=1:length(Bond_B_coe)
            bond_tensor=bond_tensor+im*B_set[ct]*Bond_B_coe[ct];
        end
    elseif Bond_irrep=="A+iB"
        bond_tensor=A_set[1]*0;
        for ct=1:length(Bond_A_coe)
            bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
        end
        for ct=1:length(Bond_B_coe)
            bond_tensor=bond_tensor+im*B_set[ct]*Bond_B_coe[ct];
        end
    end

    triangle_tensor=A1_set[1]*0;
    if Triangle_irrep=="A1"
        triangle_tensor=A1_set[1]*0;
        for ct=1:length(Triangle_A1_coe)
            triangle_tensor=triangle_tensor+A1_set[ct]*Triangle_A1_coe[ct];
        end
    elseif Triangle_irrep=="A2"
        triangle_tensor=A2_set[1]*0;
        for ct=1:length(Triangle_A2_coe)
            triangle_tensor=triangle_tensor+im*A2_set[ct]*Triangle_A2_coe[ct];
        end
    elseif Triangle_irrep=="A1+iA2"
        triangle_tensor=A1_set[1]*0;
        for ct=1:length(Triangle_A1_coe)
            triangle_tensor=triangle_tensor+A1_set[ct]*Triangle_A1_coe[ct];
        end
        for ct=1:length(Triangle_A2_coe)
            triangle_tensor=triangle_tensor+im*A2_set[ct]*Triangle_A2_coe[ct];
        end
    end



    return bond_tensor,triangle_tensor
end




function get_tensor_coes(json_dict)
    Bond_irrep=json_dict["Bond_irrep"]
    if Bond_irrep=="A"
        Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);
        Bond_B_coe=[];
    elseif Bond_irrep=="B"
        Bond_A_coe=[];
        Bond_B_coe=read_string(json_dict["coes"]["Bond_B_coe"]["entries"]);
    elseif Bond_irrep=="A+iB"
        Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);
        Bond_B_coe=read_string(json_dict["coes"]["Bond_B_coe"]["entries"]);
    end

    Triangle_irrep=json_dict["Triangle_irrep"]
    if Triangle_irrep=="A1"
        Triangle_A1_coe=read_string(json_dict["coes"]["Triangle_A1_coe"]["entries"]);
        Triangle_A2_coe=[];
    elseif Triangle_irrep=="A2"
        Triangle_A1_coe=[];
        Triangle_A2_coe=read_string(json_dict["coes"]["Triangle_A2_coe"]["entries"]);
    elseif Triangle_irrep=="A1+iA2"
        Triangle_A1_coe=read_string(json_dict["coes"]["Triangle_A1_coe"]["entries"]);
        Triangle_A2_coe=read_string(json_dict["coes"]["Triangle_A2_coe"]["entries"]);
    end

    if haskey(json_dict, "nonchiral")
        nonchiral=json_dict["nonchiral"];
    else
        nonchiral="No";
    end


    return Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe
end