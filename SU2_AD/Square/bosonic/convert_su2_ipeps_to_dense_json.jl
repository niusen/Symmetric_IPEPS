using TensorKit
using JLD2
using JSON
using LinearAlgebra: norm

cd(@__DIR__)

default_input_jld2 = "Optim_RVB_coefficients_D3_chi_54_-0.9834.jld2"
default_output_json = "Optim_RVB_coefficients_D3_chi_54_-0.9834_dense.json"

function _complex_entry_string(indices0, z::Complex)
    fields = [string(i) for i in indices0]
    push!(fields, string(real(z)))
    push!(fields, string(imag(z)))
    return join(fields, " ")
end

function _dense_entries(A::AbstractArray{<:Number}; atol::Real=0)
    entries = String[]
    for I in CartesianIndices(A)
        z = ComplexF64(A[I])
        if abs(z) > atol
            push!(entries, _complex_entry_string(Tuple(I) .- 1, z))
        end
    end
    return entries
end

function _load_ipeps_tensor(jld2file::AbstractString; tensor_key::String="A")
    data = load(jld2file)
    @assert haskey(data, tensor_key) "JLD2 file must contain tensor key `$(tensor_key)`."
    return data[tensor_key]
end

function _ipeps_to_dense_julia_array(A)
    A_dense = Array{ComplexF64}(convert(Array, A))
    @assert ndims(A_dense) == 5 "Expected a rank-5 iPEPS tensor in order (left, down, right, up, phys)."
    return A_dense
end

function su2_ipeps_jld2_to_dense_json(input_jld2::AbstractString=default_input_jld2;
        output_json::AbstractString=default_output_json,
        tensor_key::String="A",
        atol::Real=0)

    input_jld2 = abspath(input_jld2)
    output_json = abspath(output_json)
    A = _load_ipeps_tensor(input_jld2; tensor_key=tensor_key)

    # This package's square iPEPS convention is A[left, down, right, up, phys].
    A_julia = _ipeps_to_dense_julia_array(A)

    # Juraj/tn-torch convention is A[phys, up, left, down, right].
    A_juraj = permutedims(A_julia, (5, 4, 1, 2, 3))

    entries = _dense_entries(A_juraj; atol=atol)
    site = Dict(
        "siteId" => "A",
        "dtype" => "complex128",
        "format" => "legacy",
        "dims" => collect(size(A_juraj)),
        "numEntries" => length(entries),
        "entries" => entries,
    )

    state = Dict(
        "lX" => 1,
        "lY" => 1,
        "map" => [["A"]],
        "sites" => [site],
        "source_jld2" => input_jld2,
        "source_tensor_key" => tensor_key,
        "julia_leg_order" => ["left", "down", "right", "up", "phys"],
        "json_leg_order" => ["phys", "up", "left", "down", "right"],
        "dims_julia" => collect(size(A_julia)),
        "dims_json" => collect(size(A_juraj)),
        "norm_julia_dense" => norm(vec(A_julia)),
        "atol" => atol,
    )

    open(output_json, "w") do io
        JSON.print(io, state, 4)
    end

    println("Converted SU(2) iPEPS to dense JSON")
    println("  input_jld2       = " * input_jld2)
    println("  output_json      = " * output_json)
    println("  tensor_key       = " * tensor_key)
    println("  julia leg order  = " * string(state["julia_leg_order"]))
    println("  json leg order   = " * string(state["json_leg_order"]))
    println("  dims_julia       = " * string(size(A_julia)))
    println("  dims_json        = " * string(size(A_juraj)))
    println("  numEntries       = " * string(length(entries)))
    println("  norm             = " * string(norm(vec(A_julia))))
    flush(stdout)

    return output_json
end

if length(ARGS) == 0
    su2_ipeps_jld2_to_dense_json()
elseif length(ARGS) == 1
    su2_ipeps_jld2_to_dense_json(ARGS[1])
elseif length(ARGS) == 2
    su2_ipeps_jld2_to_dense_json(ARGS[1]; output_json=ARGS[2])
else
    error("Usage: julia convert_su2_ipeps_to_dense_json.jl [input_jld2] [output_json]")
end
