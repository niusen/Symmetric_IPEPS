using TensorKit
using JLD2
cd(@__DIR__)

#from version<=0.12.7 to version>0.12.7

#seems this doesn't work
function oldversion_to_Dict(t)
    t_dict = Dict(:space => space(t), :data => Dict((f₁, f₂) => t[f₁, f₂] for (f₁, f₂) in fusiontrees(t)));
    return t_dict
end

function Dict_to_newversion(t_dict)
    T = eltype(valtype(t_dict[:data]))
    t = TensorMap{T}(undef, t_dict[:space])
    for ((f₁, f₂), val) in t_dict[:data]
        t[f₁, f₂] .= val
    end
    return t
end



function tensor_to_dense_dict(t)
    t_dense=convert(Array,t);
    return Dict(:domain=>domain(t),:codomain=>codomain(t),:t_dense=>t_dense)
end

function dense_dict_to_tensor(t)
    t=TensorMap(t[:t_dense],t[:codomain],t[:domain])
    return t
end



filenm="new_SimpleUpdate_D_28.jld2";
data=load(filenm);



# for cc in eachindex(data)
#     data[cc]=oldversion_to_Dict(data[cc])
# end


# data=data["data"];
# for cc in eachindex(data)
#     data[cc]=Dict_to_newversion(data[cc])
# end



# B_a=tensor_to_dense_dict(data["B_a"]);
# B_b=tensor_to_dense_dict(data["B_b"]);
# B_c=tensor_to_dense_dict(data["B_c"]);
# T_u=tensor_to_dense_dict(data["T_u"]);
# T_d=tensor_to_dense_dict(data["T_d"]);
# jldsave("new_"*filenm;B_a=B_a,B_b=B_b,B_c=B_c,T_u=T_u,T_d=T_d);



B_a=dense_dict_to_tensor(data["B_a"]);
B_b=dense_dict_to_tensor(data["B_b"]);
B_c=dense_dict_to_tensor(data["B_c"]);
T_u=dense_dict_to_tensor(data["T_u"]);
T_d=dense_dict_to_tensor(data["T_d"]);

jldsave(filenm;B_a=B_a,B_b=B_b,B_c=B_c,T_u=T_u,T_d=T_d);
