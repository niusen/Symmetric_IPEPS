using TensorKit
using JLD2
using HDF5
using JSON

cd(@__DIR__)

function convert_to_Dict(T)
    T_dense=convert(Array,T);
    Rank=length(domain(T))+length(codomain(T));
    even_dims=Vector{Int}(undef,Rank);
    odd_dims=Vector{Int}(undef,Rank);
    dual=zeros(Int,Rank);
    for cc=1:Rank
        even_dims[cc]=space(T,cc).dims[1];
        odd_dims[cc]=space(T,cc).dims[2];
        if space(T,cc).dual
            dual[cc]=1;
        end
    end
    T_real=real(T_dense);
    T_imag=imag(T_dense);
    return Dict("T_real"=>T_real[:],"T_imag"=>T_imag[:],"even_dims"=>even_dims,"odd_dims"=>odd_dims,"dual"=>dual);
end

filenm="iPESS_Z2_D12";


T_set=Dict{String,Any}();
B_set=Dict{String,Any}();

data=load(filenm*".jld2");

    if haskey(data,"T_set")
        Lx,Ly=size(data["T_set"]);
        for cx=1:Lx
            for cy=1:Ly
                tm=data["T_set"][cx,cy];
                bm=data["B_set"][cx,cy];
                merge!(T_set,Dict{String,Any}(string(cx)*","*string(cy)=>convert_to_Dict(tm)));
                merge!(B_set,Dict{String,Any}(string(cx)*","*string(cy)=>convert_to_Dict(bm)));
            end
        end

    else
        state=data["x"];
        Lx,Ly=size(state);

        for cx=1:Lx
            for cy=1:Ly
                tm=state[cx,cy].Bm;
                bm=state[cx,cy].Tm;
                merge!(T_set,Dict{String,Any}(string(cx)*","*string(cy)=>convert_to_Dict(tm)));
                merge!(B_set,Dict{String,Any}(string(cx)*","*string(cy)=>convert_to_Dict(bm)));
            end
        end
    end







jsdata = JSON.json(Dict("T_set"=>T_set,"B_set"=>B_set))
open(filenm*".json", "w") do f
        write(f, jsdata)
     end




