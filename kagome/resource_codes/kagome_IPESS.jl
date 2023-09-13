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




function wrap_json_state(Bond_irrep,Triangle_irrep,nonchiral,Bond_A_coe,Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe)
    if Bond_irrep=="A"
        if Triangle_irrep=="A1"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe))]);
        elseif Triangle_irrep=="A2"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        elseif Triangle_irrep=="A1+iA2"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe)),("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        end
    elseif Bond_irrep=="B"
        if Triangle_irrep=="A1"
            coes=Dict([("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe))]);
        elseif Triangle_irrep=="A2"
            coes=Dict([("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        elseif Triangle_irrep=="A1+iA2"
            coes=Dict([("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe)),("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        end
    elseif Bond_irrep=="A+iB"
        if Triangle_irrep=="A1"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe))]);
        elseif Triangle_irrep=="A2"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        elseif Triangle_irrep=="A1+iA2"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe)),("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        end
    end
    json_state=Dict([("coes" , coes), ("Bond_irrep", Bond_irrep), ("Triangle_irrep", Triangle_irrep), ("nonchiral", nonchiral)]);
    
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





function initial_state(Bond_irrep,Triangle_irrep,nonchiral,D,init_statenm=nothing,init_noise=0)
    A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, _, _, virtual_particle, _, _=construct_tensor(D);
    if init_statenm=="nothing" 
        println("Random initial state");flush(stdout);
        
        if Bond_irrep=="A"
            Bond_A_coe=randn(Float64, length(A_set));
            Bond_B_coe=[];
        elseif Bond_irrep=="B"
            Bond_A_coe=[];
            Bond_B_coe=randn(Float64, length(B_set));
        elseif Bond_irrep=="A+iB"
            Bond_A_coe=randn(Float64, length(A_set));
            Bond_B_coe=randn(Float64, length(B_set));
        end
        if Triangle_irrep=="A1"
            Triangle_A1_coe=randn(Float64, length(A1_set));
            Triangle_A2_coe=[];
        elseif Triangle_irrep=="A2"
            Triangle_A1_coe=[];
            Triangle_A2_coe=randn(Float64, length(A2_set));
        elseif Triangle_irrep=="A1+iA2"
            Triangle_A1_coe=randn(Float64, length(A1_set));
            Triangle_A2_coe=randn(Float64, length(A2_set));
        end
        
        #projection to ninchiral state if needed
        Triangle_A1_coe,Triangle_A2_coe, A1_has_odd, A2_has_odd=nonchiral_projection(nonchiral,Triangle_A1_coe,Triangle_A2_coe,A1_set_occu,A2_set_occu,virtual_particle);

        json_state_dict=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe)
    else
        
        println("load state: "*init_statenm);flush(stdout);
        json_state_dict=read_json_state(init_statenm);
        Bond_irrep_, Triangle_irrep_, nonchiral_, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(json_state_dict);#projection to nonchiral is inside this function if needed 
        @assert Bond_irrep_==Bond_irrep
        @assert Triangle_irrep_==Triangle_irrep
        if nonchiral_==nonchiral
        else
            if nonchiral!="No"
                println("Initial state is not nonchiral. Now project it to nonchiral.");flush(stdout);
            end
        end

        #add initial noise
        if Bond_irrep=="A"
            Bond_A_coe=Bond_A_coe+(rand(Float64, length(Bond_A_coe)).-0.5)*init_noise;
            Bond_B_coe=[];
        elseif Bond_irrep=="B"
            Bond_A_coe=[];
            Bond_B_coe=Bond_B_coe+(rand(Float64, length(Bond_B_coe)).-0.5)*init_noise;
        elseif Bond_irrep=="A+iB"
            Bond_A_coe=Bond_A_coe+(rand(Float64, length(Bond_A_coe)).-0.5)*init_noise;
            Bond_B_coe=Bond_B_coe+(rand(Float64, length(Bond_B_coe)).-0.5)*init_noise;
        end

        if Triangle_irrep=="A1"
            Triangle_A1_coe=Triangle_A1_coe+(rand(Float64, length(Triangle_A1_coe)).-0.5)*init_noise;
            Triangle_A2_coe=[];
        elseif Triangle_irrep=="A2"
            Triangle_A1_coe=[];
            Triangle_A2_coe=Triangle_A2_coe+(rand(Float64, length(Triangle_A2_coe)).-0.5)*init_noise;
        elseif Triangle_irrep=="A1+iA2"
            Triangle_A1_coe=Triangle_A1_coe+(rand(Float64, length(Triangle_A1_coe)).-0.5)*init_noise;
            Triangle_A2_coe=Triangle_A2_coe+(rand(Float64, length(Triangle_A2_coe)).-0.5)*init_noise;
        end

        #projection to ninchiral state if needed
        Triangle_A1_coe,Triangle_A2_coe, A1_has_odd, A2_has_odd=nonchiral_projection(nonchiral,Triangle_A1_coe,Triangle_A2_coe,A1_set_occu,A2_set_occu,virtual_particle);

        #wrap the changed state due to initial noise 
        json_state_dict=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe);
    end
    return json_state_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd

end




function set_vector(json_dict, vec)
    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coes0, Bond_B_coes0, Triangle_A1_coes0, Triangle_A2_coes0=get_tensor_coes(json_dict);
    if Bond_irrep=="A"
        if Triangle_irrep=="A1"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]

            Triangle_A2_coe=nothing
        elseif Triangle_irrep=="A2"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            # siz=length(Triangle_A1_coes0)
            # Triangle_A1_coe=vec[1:siz]
            # vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]

            Triangle_A1_coe=nothing
        elseif Triangle_irrep=="A1+iA2"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]
        end

        Bond_B_coe=nothing;
    elseif Bond_irrep=="B"
        if Triangle_irrep=="A1"
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]

            Triangle_A2_coe=nothing
        elseif Triangle_irrep=="A2"
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            # siz=length(Triangle_A1_coes0)
            # Triangle_A1_coe=vec[1:siz]
            # vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]

            Triangle_A1_coe=nothing
        elseif Triangle_irrep=="A1+iA2"
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]
        end

        Bond_A_coe=nothing;
    elseif Bond_irrep=="A+iB"
        if Triangle_irrep=="A1"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]

            Triangle_A2_coe=nothing
        elseif Triangle_irrep=="A2"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            # siz=length(Triangle_A1_coes0)
            # Triangle_A1_coe=vec[1:siz]
            # vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]

            Triangle_A1_coe=nothing
        elseif Triangle_irrep=="A1+iA2"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]
        end

    end
    json_dict_new=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe)
    #return Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe
    return json_dict_new
end





function normalize_IPESS_SU2_PG(state_dict)
    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(state_dict)
    if Bond_irrep=="A"
        Bond_norm=norm(Bond_A_coe)
        Bond_A_coe=Bond_A_coe/Bond_norm
    elseif Bond_irrep=="B"
        Bond_norm=norm(Bond_B_coe)
        Bond_B_coe=Bond_B_coe/Bond_norm
    elseif Bond_irrep=="A+iB"
        Bond_norm=sqrt(norm(Bond_A_coe)^2+norm(Bond_B_coe)^2)
        Bond_A_coe=Bond_A_coe/Bond_norm
        Bond_B_coe=Bond_B_coe/Bond_norm
    end

    if Triangle_irrep=="A1"
        Triangle_norm=norm(Triangle_A1_coe)
        Triangle_A1_coe=Triangle_A1_coe/Triangle_norm
    elseif Triangle_irrep=="A2"
        Triangle_norm=norm(Triangle_A2_coe)
        Triangle_A2_coe=Triangle_A2_coe/Triangle_norm
    elseif Triangle_irrep=="A1+iA2"
        Triangle_norm=sqrt(norm(Triangle_A1_coe)^2+norm(Triangle_A2_coe)^2)
        Triangle_A1_coe=Triangle_A1_coe/Triangle_norm
        Triangle_A2_coe=Triangle_A2_coe/Triangle_norm
    end

    state_dict=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe)
    return state_dict
end



function get_vector(json_dict)
    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(json_dict);
    if Bond_irrep=="A"
        if Triangle_irrep=="A1"
            vec=vcat(Bond_A_coe,Triangle_A1_coe);
        elseif Triangle_irrep=="A2"
            vec=vcat(Bond_A_coe,Triangle_A2_coe);
        elseif Triangle_irrep=="A1+iA2"
            vec=vcat(Bond_A_coe,Triangle_A1_coe,Triangle_A2_coe);
        end
    elseif Bond_irrep=="B"
        if Triangle_irrep=="A1"
            vec=vcat(Bond_B_coe,Triangle_A1_coe);
        elseif Triangle_irrep=="A2"
            vec=vcat(Bond_B_coe,Triangle_A2_coe);
        elseif Triangle_irrep=="A1+iA2"
            vec=vcat(Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe);
        end
    elseif Bond_irrep=="A+iB"
        if Triangle_irrep=="A1"
            vec=vcat(Bond_A_coe,Bond_B_coe,Triangle_A1_coe);
        elseif Triangle_irrep=="A2"
            vec=vcat(Bond_A_coe,Bond_B_coe,Triangle_A2_coe);
        elseif Triangle_irrep=="A1+iA2"
            vec=vcat(Bond_A_coe,Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe);
        end
    end
    return vec
end