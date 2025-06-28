import Pkg
Pkg.activate(; temp = true)
Pkg.add("TensorKit@0.12.7")
#each time launch julia, need to install
]add TensorKit@0.12.7 


using TensorKit
using JLD2
cd("/home/sniu/iPEPS_new/trianlge/U8/iPESS_ES/");
data=load("stochastic_iPESS_LS_D_8_chi_120.jld2");
tt=data["T_set"][1,1]




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



# function tensor_to_dense_dict(t)
#     t_dense=convert(Array,t);
#     return Dict(:domain=>domain(t),:codomain=>codomain(t),:t_dense=>t_dense)
# end

# function dense_dict_to_tensor(t)
#     t=TensorMap(t[:t_dense],t[:codomain],t[:domain])
#     return t
# end


function tensor_to_dict(tm::Matrix)
    l1,l2=size(tm);
    tmnew=Matrix{Any}(undef,l1,l2);
    for c1 in eachindex(tm)
        tmnew[c1]=convert(Dict,tm[c1]);
    end
    return tmnew
end

function dict_to_tensor(tm::Matrix)
    l1,l2=size(tm);
    tmnew=Matrix{TensorMap}(undef,l1,l2);
    for c1 in eachindex(tm)    
        tmnew[c1]=convert(TensorMap,tm[c1]);
    end
    return tmnew
end

#save 
filenm="stochastic_iPESS_LS_D_8_chi_120.jld2";
data=load(filenm);
T_set=data["T_set"];
B_set=data["B_set"];

# #for triangle_iPESS structure 
# x=data["x"];
# T_set=Matrix{TensorMap}(undef,2,2);
# B_set=Matrix{TensorMap}(undef,2,2);
# for cx=1:2
# for cy=1:2
# T_set[cx,cy]=x[cx,cy].Bm;
# B_set[cx,cy]=x[cx,cy].Tm;
# end
# end

T_set=tensor_to_dict(T_set);
B_set=tensor_to_dict(B_set);
jldsave("newversion_"*filenm;T_set=T_set,B_set=B_set);

######################

#load 
data=load("newversion_"*filenm);
T_set=data["T_set"];
B_set=data["B_set"];

T_set=dict_to_tensor(T_set);
B_set=dict_to_tensor(B_set);
jldsave("newnewversion_"*filenm;T_set=T_set,B_set=B_set);