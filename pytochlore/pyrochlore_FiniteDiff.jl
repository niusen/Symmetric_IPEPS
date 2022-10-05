
function read_json_state(filenm)
    json_dict = Dict()
    open(filenm, "r") do f
        json_dict
        dicttxt = read(f,String)  # file information to string
        json_dict=JSON.parse(dicttxt)  # parse and transform data
    end
    return json_dict
end


function wrap_json_state(Bond_irrep,Tetrahedral_irrep,Bond_A_coe,Tetrahedral_E_coe)
    coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Tetrahedral_E_coe", create_coe_dict(Tetrahedral_E_coe))]);
    json_state=Dict([("coes" , coes), ("Bond_irrep", Bond_irrep), ("Tetrahedral_irrep", Tetrahedral_irrep)]);
    return json_state
end


function create_coe_dict(coe)
    #print(coe)
    entries=Vector(undef,length(coe));
    for cc=1:length(coe)
        entries[cc]=string(cc-1)*" "*string(coe[cc]);
    end
    dims=Vector(undef,1);
    dims[1]=length(coe);

    coe_dict=Dict([("dtype", "float64"), ("numEntries", length(coe)),("entries", entries), ("dims", dims)]);
    return coe_dict
end

function initial_state(Bond_irrep,Tetrahedral_irrep,D,init_statenm=nothing,init_noise=0)
    A_set,E_set,S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
    if init_statenm==nothing 
        println("Random initial state");flush(stdout);
        

        Bond_A_coe=randn(Float64, length(A_set));
        Tetrahedral_E_coe=randn(Float64, length(E_set));

        json_state_dict=wrap_json_state(Bond_irrep, Tetrahedral_irrep, Bond_A_coe, Tetrahedral_E_coe)
    else
        
        println("load state: "*init_statenm);flush(stdout);
        json_state_dict=read_json_state(init_statenm);
        Bond_irrep_, Tetrahedral_irrep_, Bond_A_coe, Tetrahedral_E_coe=get_tensor_coes(json_state_dict);#projection to nonchiral is inside this function if needed 
        @assert Bond_irrep_==Bond_irrep
        @assert Tetrahedral_irrep_==Tetrahedral_irrep

        #add initial noise
        Bond_A_coe=Bond_A_coe+(rand(Float64, length(Bond_A_coe)).-0.5)*init_noise;
        Tetrahedral_E_coe=Tetrahedral_E_coe+(rand(Float64, length(Tetrahedral_E_coe)).-0.5)*init_noise;

        #wrap the changed state due to initial noise 
        json_state_dict=wrap_json_state(Bond_irrep, Tetrahedral_irrep, Bond_A_coe, Tetrahedral_E_coe);
    end
    return json_state_dict, Bond_A_coe, Tetrahedral_E_coe

end



function get_vector(json_dict)
    Bond_irrep, Tetrahedral_irrep, Bond_A_coe, Tetrahedral_E_coe=get_tensor_coes(json_dict); 
    vec=vcat(Bond_A_coe,Tetrahedral_E_coe);
    return vec
end


function set_vector(json_dict, vec)
    Bond_irrep, Tetrahedral_irrep, Bond_A_coes0, Tetrahedral_E_coes0=get_tensor_coes(json_dict);

    siz=length(Bond_A_coes0)
    Bond_A_coe=vec[1:siz]
    vec=vec[siz+1:length(vec)]
    siz=length(Tetrahedral_E_coes0)
    Tetrahedral_E_coe=vec[1:siz]

    json_dict_new=wrap_json_state(Bond_irrep, Tetrahedral_irrep, Bond_A_coe, Tetrahedral_E_coe)
    return json_dict_new
end

