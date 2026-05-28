using JSON
using JLD2
using LinearAlgebra: norm
using TensorKit

const DEFAULT_JURAJ_IPEPS_JSON = raw"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev\data_c4_pt_csl\j1j2lambda_c4pt_D3_chi40_seed0_gpu_state_-0.9865.json"

function _resolve_juraj_json_path(jsonfile::AbstractString)
    if isfile(jsonfile)
        return jsonfile
    elseif isfile(jsonfile * ".json")
        return jsonfile * ".json"
    end
    error("Could not find Juraj iPEPS JSON file: $(jsonfile)")
end

function _parse_number_token(x, dtype_str::AbstractString)
    if x isa Number
        return ComplexF64(x, 0)
    elseif x isa AbstractDict
        return ComplexF64(Float64(x["real"]), Float64(x["imag"]))
    end

    s = strip(String(x))
    if occursin("complex", lowercase(dtype_str)) || occursin("j", s) || occursin("im", s)
        s = replace(s, "j" => "im")
        return parse(ComplexF64, s)
    else
        return ComplexF64(parse(Float64, s), 0)
    end
end

function _reshape_c_order(v::AbstractVector, dims::Vector{Int})
    # Python/torch flattening uses C order: last index moves fastest.
    # Julia reshape uses Fortran order, so reverse dimensions then permute back.
    nd = length(dims)
    return permutedims(reshape(collect(v), reverse(dims)...), reverse(1:nd))
end

function _read_bare_json_tensor(site::Dict)
    dtype_str = lowercase(get(site, "dtype", "float64"))
    dims = Int.(site["dims"])

    if get(site, "format", "legacy") == "1D"
        raw = [_parse_number_token(x, dtype_str) for x in site["data"]]
        @assert length(raw) == prod(dims) "1D tensor data length does not match dims."
        return Array{ComplexF64}(_reshape_c_order(raw, dims))
    end

    A = zeros(ComplexF64, dims...)
    nd = length(dims)
    for entry in site["entries"]
        fields = split(entry)
        inds = parse.(Int, fields[1:nd]) .+ 1
        if occursin("complex", dtype_str)
            val = parse(Float64, fields[nd + 1]) + im * parse(Float64, fields[nd + 2])
        else
            val = parse(Float64, fields[nd + 1])
        end
        A[inds...] = val
    end
    return A
end

function _find_site(raw_state::Dict; site_id=nothing, site_index::Int=1)
    sites = raw_state["sites"]
    if site_id === nothing
        @assert 1 <= site_index <= length(sites) "site_index is out of range."
        return sites[site_index]
    end
    for site in sites
        if site["siteId"] == site_id
            return site
        end
    end
    error("Could not find siteId=$(site_id).")
end

function read_juraj_ipeps_tensor(jsonfile::AbstractString=DEFAULT_JURAJ_IPEPS_JSON;
        site_id=nothing, site_index::Int=1)
    jsonfile = _resolve_juraj_json_path(jsonfile)
    raw_state = JSON.parsefile(jsonfile)
    site = _find_site(raw_state; site_id=site_id, site_index=site_index)

    A_python = _read_bare_json_tensor(site)

    # Juraj/tn-torch IPEPS convention is a[s,u,l,d,r].
    # This repository's square TensorMap convention is A[l,d,r,u,s].
    A_julia = permutedims(A_python, (3, 4, 5, 2, 1))

    return (
        A_python=A_python,
        A_julia=A_julia,
        python_leg_order=("phys", "up", "left", "down", "right"),
        julia_leg_order=("left", "down", "right", "up", "phys"),
        jsonfile=jsonfile,
        site_id=site["siteId"],
        dims_python=size(A_python),
        dims_julia=size(A_julia),
        lX=raw_state["lX"],
        lY=raw_state["lY"],
        map=raw_state["map"],
    )
end

function juraj_ipeps_array_to_tensormap(A_julia::AbstractArray{<:Number,5})
    Dl, Dd, Dr, Du, dp = size(A_julia)
    A = TensorMap(
        Array{ComplexF64}(A_julia),
        ComplexSpace(Dl) * ComplexSpace(Dd) * ComplexSpace(Dr)' * ComplexSpace(Du)',
        ComplexSpace(dp),
    )
    # The square CTMRG code contracts A and A' as rank-5 tensors, matching
    # initial_dense_state in square_AD_SU2.jl.
    return permute(A, (1, 2, 3, 4, 5,))
end

function read_juraj_ipeps_tensormap(jsonfile::AbstractString=DEFAULT_JURAJ_IPEPS_JSON;
        site_id=nothing, site_index::Int=1)
    data = read_juraj_ipeps_tensor(jsonfile; site_id=site_id, site_index=site_index)
    return juraj_ipeps_array_to_tensormap(data.A_julia)
end

function save_juraj_ipeps_tensor_jld2(jsonfile::AbstractString=DEFAULT_JURAJ_IPEPS_JSON;
        outputfile=nothing, site_id=nothing, site_index::Int=1)
    jsonfile = _resolve_juraj_json_path(jsonfile)
    data = read_juraj_ipeps_tensor(jsonfile; site_id=site_id, site_index=site_index)
    if outputfile === nothing
        outputfile = replace(basename(jsonfile), ".json" => "_julia_order.jld2")
    end
    A_tensormap = juraj_ipeps_array_to_tensormap(data.A_julia)
    jldsave(
        outputfile;
        A_julia=data.A_julia,
        A_tensormap=A_tensormap,
        A_python=data.A_python,
        python_leg_order=collect(data.python_leg_order),
        julia_leg_order=collect(data.julia_leg_order),
        jsonfile=data.jsonfile,
        site_id=data.site_id,
        dims_python=collect(data.dims_python),
        dims_julia=collect(data.dims_julia),
        lX=data.lX,
        lY=data.lY,
    )
    return outputfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    data = read_juraj_ipeps_tensor(DEFAULT_JURAJ_IPEPS_JSON)
    println("Read Juraj iPEPS tensor from: " * data.jsonfile)
    println("site_id = " * string(data.site_id))
    println("python leg order = " * string(data.python_leg_order))
    println("python dims = " * string(data.dims_python))
    println("julia leg order = " * string(data.julia_leg_order))
    println("julia dims = " * string(data.dims_julia))
    println("norm(A_python) = " * string(norm(vec(data.A_python))))
    println("norm(A_julia) = " * string(norm(vec(data.A_julia))))
    println("TensorMap spaces = " * string(space(read_juraj_ipeps_tensormap(DEFAULT_JURAJ_IPEPS_JSON))))
    out = save_juraj_ipeps_tensor_jld2(DEFAULT_JURAJ_IPEPS_JSON)
    println("Saved Julia-order dense tensor to " * out)
end
