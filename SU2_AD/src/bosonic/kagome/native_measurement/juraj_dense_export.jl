# Exact file conversion only. Numerical measurements are performed by Juraj.
if !isdefined(@__MODULE__,:KagomeNativeMeasurement)
    include(joinpath(@__DIR__,"KagomeNativeMeasurement.jl"))
end
module KagomeJurajDenseExport
using TOML, SHA, LinearAlgebra
import ..KagomeNativeMeasurement
const Native=KagomeNativeMeasurement

function source_data(path)
    if endswith(lowercase(path),".toml")
        d=TOML.parsefile(path)
        d["format"]=="kagome_u1_portable_v1" || error("expected the final U1 u1_state.toml portable file")
        d["charge_unit"]=="2Sz" && d["cell_pattern"]=="sqrt3xsqrt3" || error("expected the sqrt3 U1 state")
        return d
    end
    state,_=Native.KagomeU1Cells.load_u1_checkpoint(path)
    Native.KagomeU1Cells.export_u1_portable(state)
end

function decode_tensor(item)
    A=reshape(ComplexF64.(item["real"],item["imag"]),Tuple(item["dims"]))
    ndims(A)==3 && isfinite(norm(A)) && norm(A)>0 || error("expected a nonzero rank-3 projected PESS tensor")
    bad=0.
    for i in CartesianIndices(A)
        q=sum(item["signatures"][k]*item["axis_charges"][k][i[k]] for k in 1:3)
        q==item["total_charge"] || (bad+=abs2(A[i]))
    end
    sqrt(bad)<=1e-12*max(norm(A),1) || error("portable U1 charges and coefficients disagree")
    A
end

function legacy_dense_tensor(A)
    # Explicit indices make Julia column-major / Python row-major irrelevant.
    # Keep every nonzero coefficient; no numerical truncation is applied.
    entries=String[]
    for i in CartesianIndices(A)
        v=A[i];iszero(v) && continue
        push!(entries,join((string(i[k]-1) for k in 1:ndims(A))," ")*" "*string(real(v))*" "*string(imag(v)))
    end
    Dict("dtype"=>"complex128","dims"=>collect(size(A)),"entries"=>entries)
end

function export_juraj_dense(source,target;D=0)
    data=source_data(source)
    data["Lx"]==data["Ly"]==3 && length(data["cells"])==3 || error("expected three independent PESS tensors on the 3x3 display")
    tensors=Dict{String,Any}();inverse_error=0.;actual=Int[]
    for r in 1:3
        c=data["cells"][string(r)]
        values=Dict(k=>decode_tensor(c[k]) for k in ("B1","B2","B3","Tup","Tdn"))
        for k in ("B1","B2","B3")
            c[k]["axis_charges"][3]==[1,-1] || error("physical basis is not up,down")
            append!(actual,size(values[k])[1:2])
        end
        mapped=Dict("T_u"=>permutedims(values["Tdn"],(2,3,1)),
            "T_d"=>permutedims(values["Tup"],(2,3,1)),
            "B_a"=>permutedims(values["B1"],(3,2,1)),
            "B_b"=>permutedims(values["B3"],(3,2,1)),
            "B_c"=>permutedims(values["B2"],(3,2,1)))
        for (name,key,perm) in (("T_u","Tdn",(3,1,2)),("T_d","Tup",(3,1,2)),
                ("B_a","B1",(3,2,1)),("B_b","B3",(3,2,1)),("B_c","B2",(3,2,1)))
            inverse_error=max(inverse_error,norm(permutedims(mapped[name],perm)-values[key]))
        end
        tensors["(0, $(r-1))"]=Dict(k=>legacy_dense_tensor(A) for (k,A) in mapped)
    end
    D==0 || maximum(actual)==D || error("requested D=$D, actual maximum bond dimension=$(maximum(actual))")
    pattern=[["(0, $(mod(y-x,3)))" for x in 0:2] for y in 0:2]
    metadata=Dict("source"=>abspath(source),"source_sha256"=>bytes2hex(sha256(read(source))),
        "source_format"=>"projected Julia U1 PESS","representation"=>"dense complex128",
        "D"=>maximum(actual),"inverse_permutation_error"=>inverse_error,
        "coordinate_map"=>"Juraj (X,Y)=(1-x,1-y) for one-based Julia (x,y); 180 degree rotation",
        "physical_order"=>"Juraj (s0,s1,s2)=(Julia B2,B3,B1)",
        "julia_representatives"=>Dict("(0, $(r-1))"=>"$r,1" for r in 1:3),
        "projection"=>"U1 input already has the auxiliary component selected; remove only its one-dimensional spectator")
    output=Dict("lX"=>3,"lY"=>3,"pattern"=>pattern,"ipess_tensors"=>tensors,"metadata"=>metadata)
    mkpath(dirname(target));Native.write_json(target,output)
    println("EXPORTED D=",maximum(actual)," U1 -> Juraj dense JSON: ",abspath(target));flush(stdout)
    metadata
end
end
