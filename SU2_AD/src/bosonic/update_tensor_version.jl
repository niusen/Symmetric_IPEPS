using TensorKit
using JLD2

#from version<=0.12.7 to version>0.12.7

#seems this doesn't work
function oldversion_to_Dict(t)
    t_dict = Dict(:space => space(t), :data => Dict((f₁, f₂) => t[f₁, f₂] for (f₁, f₂) in fusiontrees(t)));
    return t_dict
end

function Dict_to_newversion(t)
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