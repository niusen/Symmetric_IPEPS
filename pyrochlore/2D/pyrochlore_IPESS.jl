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


function construct_su2_PG_IPESS(json_dict,A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb)
    Bond_irrep=json_dict["Bond_irrep"]

    Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);

    Square_irrep=json_dict["Square_irrep"]
    if Square_irrep=="A1"
        Square_A1_coe=read_string(json_dict["coes"]["Square_A1_coe"]["entries"]);
    elseif Square_irrep=="A2"
        Square_A2_coe=read_string(json_dict["coes"]["Square_A2_coe"]["entries"]);
    elseif Square_irrep=="B1"
        Square_B1_coe=read_string(json_dict["coes"]["Square_B1_coe"]["entries"]);
    elseif Square_irrep=="B2"
        Square_B2_coe=read_string(json_dict["coes"]["Square_B2_coe"]["entries"]);
    end




    #combine tensors with coefficients
    bond_tensor=A_set[1]*0;
    if Bond_irrep=="A"
        for ct=1:length(Bond_A_coe)
            bond_tensor=bond_tensor+A_set[ct]*Bond_A_coe[ct];
        end
    elseif Bond_irrep=="B"
        for ct=1:length(Bond_B_coe)
            bond_tensor=bond_tensor+B_set[ct]*Bond_B_coe[ct];
        end
    end

    Square_tensor=A1_set[1]*0;
    if Square_irrep=="A1"
        for ct=1:length(Square_A1_coe)
            Square_tensor=Square_tensor+A1_set[ct]*Square_A1_coe[ct];
        end
    elseif Square_irrep=="B1"
        for ct=1:length(Square_B1_coe)
            Square_tensor=Square_tensor+B1_set[ct]*Square_B1_coe[ct];
        end
    elseif Square_irrep=="A2"
        for ct=1:length(Square_A2_coe)
            Square_tensor=Square_tensor+A2_set[ct]*Square_A2_coe[ct];
        end
    elseif Square_irrep=="B2"
        for ct=1:length(Square_B2_coe)
            Square_tensor=Square_tensor+B2_set[ct]*Square_B2_coe[ct];
        end
    end

    return bond_tensor,Square_tensor
end




function get_tensor_coes(json_dict)
    Bond_irrep=json_dict["Bond_irrep"]
    Bond_A_coe=read_string(json_dict["coes"]["Bond_A_coe"]["entries"]);

    Square_irrep=json_dict["Square_irrep"]
    Square_A1_coe=read_string(json_dict["coes"]["Square_A1_coe"]["entries"]);
    Square_A2_coe=read_string(json_dict["coes"]["Square_A2_coe"]["entries"]);
    Square_B1_coe=read_string(json_dict["coes"]["Square_B1_coe"]["entries"]);
    Square_B2_coe=read_string(json_dict["coes"]["Square_B2_coe"]["entries"]);

    return Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe
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




function wrap_json_state(Bond_irrep, Square_irrep,Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe);
    coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Square_A1_coe", create_coe_dict(Square_A1_coe)), ("Square_A2_coe", create_coe_dict(Square_A2_coe)), ("Square_B1_coe", create_coe_dict(Square_B1_coe)), ("Square_B2_coe", create_coe_dict(Square_B2_coe))]);
    json_state=Dict([("coes" , coes), ("Bond_irrep", Bond_irrep), ("Square_irrep", Square_irrep)]);
    return json_state
end

# function wrap_json_state(Bond_irrep,Triangle_irrep,nonchiral,Bond_A_coe,Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe)
#     if Bond_irrep=="A"
#         if Triangle_irrep=="A1"
#             coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe))]);
#         elseif Triangle_irrep=="A2"
#             coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
#         elseif Triangle_irrep=="A1+iA2"
#             coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe)),("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
#         end
#     elseif Bond_irrep=="B"
#         if Triangle_irrep=="A1"
#             coes=Dict([("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe))]);
#         elseif Triangle_irrep=="A2"
#             coes=Dict([("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
#         elseif Triangle_irrep=="A1+iA2"
#             coes=Dict([("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe)),("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
#         end
#     elseif Bond_irrep=="A+iB"
#         if Triangle_irrep=="A1"
#             coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe))]);
#         elseif Triangle_irrep=="A2"
#             coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
#         elseif Triangle_irrep=="A1+iA2"
#             coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe)),("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
#         end
#     end
#     json_state=Dict([("coes" , coes), ("Bond_irrep", Bond_irrep), ("Triangle_irrep", Triangle_irrep), ("nonchiral", nonchiral)]);
    
#     return json_state
# end


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



function initial_state(Bond_irrep,Square_irrep,D,init_statenm,init_noise)
    global A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb
    if init_statenm=="nothing" 
        println("Random initial state");flush(stdout);
        
        if Bond_irrep=="A"
            Bond_A_coe=randn(Float64, length(A_set));
            Bond_B_coe=[];
        elseif Bond_irrep=="B"
            Bond_A_coe=[];
            Bond_B_coe=randn(Float64, length(B_set));
        end

        if Square_irrep=="A1"
            Square_A1_coe=randn(Float64, length(A1_set));
            Square_A2_coe=[];
            Square_B1_coe=[];
            Square_B2_coe=[];
        elseif Square_irrep=="A2"
            Square_A1_coe=[];
            Square_A2_coe=randn(Float64, length(A2_set));
            Square_B1_coe=[];
            Square_B2_coe=[];
        elseif Square_irrep=="B1"
            Square_A1_coe=[];
            Square_A2_coe=[];
            Square_B1_coe=randn(Float64, length(B1_set));
            Square_B2_coe=[];
        elseif Square_irrep=="B2"
            Square_A1_coe=[];
            Square_A2_coe=[];
            Square_B1_coe=[];
            Square_B2_coe=randn(Float64, length(B2_set));
        end

        json_state_dict=wrap_json_state(Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe);
    else
        
        println("load state: "*init_statenm);flush(stdout);
        json_state_dict=read_json_state(init_statenm);
        Bond_irrep_, Square_irrep_, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=get_tensor_coes(json_state_dict);
        @assert Bond_irrep_==Bond_irrep
        @assert Square_irrep_==Square_irrep

        #add initial noise
        if Bond_irrep=="A"
            Bond_A_coe=Bond_A_coe+(rand(Float64, length(Bond_A_coe)).-0.5)*init_noise;
        end

        if Square_irrep=="A1"
            Square_A1_coe=Square_A1_coe+(rand(Float64, length(Square_A1_coe)).-0.5)*init_noise;
        elseif Square_irrep=="A2"
            Square_A2_coe=Square_A2_coe+(rand(Float64, length(Square_A2_coe)).-0.5)*init_noise;
        elseif Square_irrep=="B1"
            Square_B1_coe=Square_B1_coe+(rand(Float64, length(Square_B1_coe)).-0.5)*init_noise;
        elseif Square_irrep=="B2"
            Square_B2_coe=Square_B2_coe+(rand(Float64, length(Square_B2_coe)).-0.5)*init_noise;
        end

        #wrap the changed state due to initial noise 
        json_state_dict=wrap_json_state(Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe);
    end
    return json_state_dict, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe

end


function get_vector(json_dict)
    Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=get_tensor_coes(json_dict); 
    if Bond_irrep=="A"
        if Square_irrep=="A1"
            vec=vcat(Bond_A_coe,Square_A1_coe);
        elseif Square_irrep=="A2"
            vec=vcat(Bond_A_coe,Square_A2_coe);
        elseif Square_irrep=="B1"
            vec=vcat(Bond_A_coe,Square_B1_coe);
        elseif Square_irrep=="B2"
            vec=vcat(Bond_A_coe,Square_B2_coe);
        end
    end
    return vec
end

# function get_vector(json_dict)
#     Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(json_dict);
#     if Bond_irrep=="A"
#         if Triangle_irrep=="A1"
#             vec=vcat(Bond_A_coe,Triangle_A1_coe);
#         elseif Triangle_irrep=="A2"
#             vec=vcat(Bond_A_coe,Triangle_A2_coe);
#         elseif Triangle_irrep=="A1+iA2"
#             vec=vcat(Bond_A_coe,Triangle_A1_coe,Triangle_A2_coe);
#         end
#     elseif Bond_irrep=="B"
#         if Triangle_irrep=="A1"
#             vec=vcat(Bond_B_coe,Triangle_A1_coe);
#         elseif Triangle_irrep=="A2"
#             vec=vcat(Bond_B_coe,Triangle_A2_coe);
#         elseif Triangle_irrep=="A1+iA2"
#             vec=vcat(Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe);
#         end
#     elseif Bond_irrep=="A+iB"
#         if Triangle_irrep=="A1"
#             vec=vcat(Bond_A_coe,Bond_B_coe,Triangle_A1_coe);
#         elseif Triangle_irrep=="A2"
#             vec=vcat(Bond_A_coe,Bond_B_coe,Triangle_A2_coe);
#         elseif Triangle_irrep=="A1+iA2"
#             vec=vcat(Bond_A_coe,Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe);
#         end
#     end
#     return vec
# end



function set_vector(json_dict, vec)
    Bond_irrep, Square_irrep, Bond_A_coe0, Square_A1_coe0, Square_A2_coe0, Square_B1_coe0, Square_B2_coe0=get_tensor_coes(json_dict);

    if Bond_irrep=="A"
        siz=length(Bond_A_coe0)
        Bond_A_coe=vec[1:siz]
        vec=vec[siz+1:length(vec)]
        if Square_irrep=="A1"
            siz=length(Square_A1_coe0)
            Square_A1_coe=vec[1:siz]
            Square_A2_coe=[];
            Square_B1_coe=[];
            Square_B2_coe=[];
        elseif Square_irrep=="A2"
            siz=length(Square_A2_coe0)
            Square_A2_coe=vec[1:siz]
            Square_A1_coe=[];
            Square_B1_coe=[];
            Square_B2_coe=[];
        elseif Square_irrep=="B1"
            siz=length(Square_B1_coe0)
            Square_B1_coe=vec[1:siz]
            Square_B2_coe=[];
            Square_A1_coe=[];
            Square_A2_coe=[];
        elseif Square_irrep=="B2"
            siz=length(Square_B2_coe0)
            Square_B2_coe=vec[1:siz]
            Square_B1_coe=[];
            Square_A1_coe=[];
            Square_A2_coe=[];
        end

    end

    json_dict_new=wrap_json_state(Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe);
    return json_dict_new
end



# function set_vector(json_dict, vec)
#     Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coes0, Bond_B_coes0, Triangle_A1_coes0, Triangle_A2_coes0=get_tensor_coes(json_dict);
#     if Bond_irrep=="A"
#         if Triangle_irrep=="A1"
#             siz=length(Bond_A_coes0)
#             Bond_A_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A1_coes0)
#             Triangle_A1_coe=vec[1:siz]

#             Triangle_A2_coe=nothing
#         elseif Triangle_irrep=="A2"
#             siz=length(Bond_A_coes0)
#             Bond_A_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             # siz=length(Triangle_A1_coes0)
#             # Triangle_A1_coe=vec[1:siz]
#             # vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A2_coes0)
#             Triangle_A2_coe=vec[1:siz]

#             Triangle_A1_coe=nothing
#         elseif Triangle_irrep=="A1+iA2"
#             siz=length(Bond_A_coes0)
#             Bond_A_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A1_coes0)
#             Triangle_A1_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A2_coes0)
#             Triangle_A2_coe=vec[1:siz]
#         end

#         Bond_B_coe=nothing;
#     elseif Bond_irrep=="B"
#         if Triangle_irrep=="A1"
#             siz=length(Bond_B_coes0)
#             Bond_B_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A1_coes0)
#             Triangle_A1_coe=vec[1:siz]

#             Triangle_A2_coe=nothing
#         elseif Triangle_irrep=="A2"
#             siz=length(Bond_B_coes0)
#             Bond_B_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             # siz=length(Triangle_A1_coes0)
#             # Triangle_A1_coe=vec[1:siz]
#             # vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A2_coes0)
#             Triangle_A2_coe=vec[1:siz]

#             Triangle_A1_coe=nothing
#         elseif Triangle_irrep=="A1+iA2"
#             siz=length(Bond_B_coes0)
#             Bond_B_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A1_coes0)
#             Triangle_A1_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A2_coes0)
#             Triangle_A2_coe=vec[1:siz]
#         end

#         Bond_A_coe=nothing;
#     elseif Bond_irrep=="A+iB"
#         if Triangle_irrep=="A1"
#             siz=length(Bond_A_coes0)
#             Bond_A_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Bond_B_coes0)
#             Bond_B_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A1_coes0)
#             Triangle_A1_coe=vec[1:siz]

#             Triangle_A2_coe=nothing
#         elseif Triangle_irrep=="A2"
#             siz=length(Bond_A_coes0)
#             Bond_A_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Bond_B_coes0)
#             Bond_B_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             # siz=length(Triangle_A1_coes0)
#             # Triangle_A1_coe=vec[1:siz]
#             # vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A2_coes0)
#             Triangle_A2_coe=vec[1:siz]

#             Triangle_A1_coe=nothing
#         elseif Triangle_irrep=="A1+iA2"
#             siz=length(Bond_A_coes0)
#             Bond_A_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Bond_B_coes0)
#             Bond_B_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A1_coes0)
#             Triangle_A1_coe=vec[1:siz]
#             vec=vec[siz+1:length(vec)]
#             siz=length(Triangle_A2_coes0)
#             Triangle_A2_coe=vec[1:siz]
#         end

#     end
#     json_dict_new=wrap_json_state(Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe,)
#     #return Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe
#     return json_dict_new
# end


function normalize_IPESS_SU2_PG(state_dict)
    Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=get_tensor_coes(state_dict);

    if Bond_irrep=="A"
        Bond_norm=norm(Bond_A_coe)
        Bond_A_coe=Bond_A_coe/Bond_norm
    # elseif Bond_irrep=="B"
    #     Bond_norm=norm(Bond_B_coe)
    #     Bond_B_coe=Bond_B_coe/Bond_norm
    # elseif Bond_irrep=="A+iB"
    #     Bond_norm=sqrt(norm(Bond_A_coe)^2+norm(Bond_B_coe)^2)
    #     Bond_A_coe=Bond_A_coe/Bond_norm
    #     Bond_B_coe=Bond_B_coe/Bond_norm
    end

    if Square_irrep=="A1"
        Square_norm=norm(Square_A1_coe)
        Square_A1_coe=Square_A1_coe/Square_norm
    elseif Square_irrep=="A2"
        Square_norm=norm(Square_A2_coe)
        Square_A2_coe=Square_A2_coe/Square_norm
    elseif Square_irrep=="B1"
        Square_norm=norm(Square_B1_coe)
        Square_B1_coe=Square_B1_coe/Square_norm
    elseif Square_irrep=="B2"
        Square_norm=norm(Square_B2_coe)
        Square_B2_coe=Square_B2_coe/Square_norm
    end

    state_dict=wrap_json_state(Bond_irrep, Square_irrep,Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe)
    return state_dict
end



