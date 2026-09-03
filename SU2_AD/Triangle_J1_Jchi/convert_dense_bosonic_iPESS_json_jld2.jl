using TensorKit
using JLD2
using JSON
using LinearAlgebra: norm

if !isdefined(@__MODULE__, :Triangle_iPESS)
    include("../src/bosonic/iPEPS_ansatz.jl")
end

const DENSE_BOSONIC_IPESS_FORMAT = "dense_bosonic_triangle_iPESS_v1"
const B_SIGNATURES = [1, 1, -1]
const T_SIGNATURES = [1, 1, -1, -1]


function _read_json_dictionary(filename::AbstractString)
    open(filename, "r") do io
        return JSON.parse(read(io, String))
    end
end


function _cell_coordinate(key::AbstractString)
    fields = split(key, ",")
    length(fields) == 2 || error("Invalid unit-cell key `$key`; expected `cx,cy`.")
    coordinate = try
        parse.(Int, fields)
    catch
        error("Invalid unit-cell key `$key`; coordinates must be integers.")
    end
    all(>(0), coordinate) || error("Unit-cell coordinates must start at 1; got `$key`.")
    return Tuple(coordinate)
end


function _cell_size(B_dictionary, T_dictionary)
    B_keys = Set(String.(keys(B_dictionary)))
    T_keys = Set(String.(keys(T_dictionary)))
    B_keys == T_keys || error(
        "B_set and T_set contain different unit-cell keys: " *
        "only_B=$(sort!(collect(setdiff(B_keys, T_keys)))), " *
        "only_T=$(sort!(collect(setdiff(T_keys, B_keys))))."
    )
    isempty(B_keys) && error("The JSON state contains an empty unit cell.")

    coordinates = _cell_coordinate.(collect(B_keys))
    Lx = maximum(first, coordinates)
    Ly = maximum(last, coordinates)
    expected = Set("$cx,$cy" for cx in 1:Lx for cy in 1:Ly)
    B_keys == expected || error(
        "Unit-cell keys do not form a complete 1:$Lx by 1:$Ly rectangle; " *
        "missing=$(sort!(collect(setdiff(expected, B_keys))))."
    )
    return Lx, Ly
end


function _decode_fortran_array(entry, tensor_name::AbstractString, expected_rank::Integer)
    haskey(entry, "dims") || error(
        "$tensor_name has no `dims` field. This converter accepts the dense " *
        "Python format, not the older Z2 `even_dims`/`odd_dims` format."
    )
    dims = Int.(entry["dims"])
    length(dims) == expected_rank || error(
        "$tensor_name has rank $(length(dims)); expected rank $expected_rank."
    )
    all(>(0), dims) || error("$tensor_name has non-positive dimensions $dims.")
    haskey(entry, "T_real") || error("$tensor_name has no `T_real` field.")
    haskey(entry, "T_imag") || error("$tensor_name has no `T_imag` field.")
    real_part = Float64.(entry["T_real"])
    imag_part = Float64.(entry["T_imag"])
    expected_length = prod(dims)
    length(real_part) == expected_length || error(
        "$tensor_name has $(length(real_part)) real entries; expected $expected_length from dims=$dims."
    )
    length(imag_part) == expected_length || error(
        "$tensor_name has $(length(imag_part)) imaginary entries; expected $expected_length from dims=$dims."
    )

    # Python writes `numpy.reshape(array, -1, order="F")`. Julia arrays are
    # column-major, so this reshape is the exact inverse: no permutation and
    # no C-order conversion is required.
    return reshape(complex.(real_part, imag_part), Tuple(dims))
end


function _check_signatures(entry, expected, tensor_name::AbstractString)
    haskey(entry, "signatures") || return
    signatures = Int.(entry["signatures"])
    signatures == expected || error(
        "$tensor_name has signatures=$signatures; expected $expected for the " *
        "Python dense triangular iPESS convention."
    )
end


function _array_to_simplex_tensor(array::AbstractArray{<:Number,3})
    dL, dU, dM = size(array)
    VL, VU, VM = ComplexSpace(dL), ComplexSpace(dU), ComplexSpace(dM)
    # Python B=(L,U,M), signatures=(+,+,-).
    return TensorMap(Array(array), VL ⊗ VU, dual(VM))
end


function _array_to_site_tensor(array::AbstractArray{<:Number,4})
    dM, ds, dR, dD = size(array)
    VM, Vp = ComplexSpace(dM), ComplexSpace(ds)
    VR, VD = ComplexSpace(dR), ComplexSpace(dD)
    # Repository convention is T=(M,s,R,D), represented as M <- (s,R,D).
    # The open physical leg is bent into the domain in the Julia TensorMap.
    return TensorMap(Array(array), dual(VM), dual(Vp) ⊗ VR ⊗ VD)
end


function _validate_dense_ipess_dimensions(B_set, T_set)
    virtual_dimensions = Set{Int}()
    physical_dimensions = Set{Int}()
    for index in eachindex(B_set, T_set)
        B_dims = size(convert(Array, B_set[index]))
        T_dims = size(convert(Array, T_set[index]))
        length(B_dims) == 3 || error("B_set[$index] is not rank 3.")
        length(T_dims) == 4 || error("T_set[$index] is not rank 4.")
        union!(virtual_dimensions, B_dims)
        union!(virtual_dimensions, (T_dims[1], T_dims[3], T_dims[4]))
        push!(physical_dimensions, T_dims[2])
    end
    length(virtual_dimensions) == 1 || error(
        "Full update requires one common dense virtual dimension; found " *
        "$(sort!(collect(virtual_dimensions)))."
    )
    length(physical_dimensions) == 1 || error(
        "Physical dimensions differ across the cell: $(sort!(collect(physical_dimensions)))."
    )
    return only(virtual_dimensions), only(physical_dimensions)
end


"""
    dense_bosonic_ipess_json_to_jld2(input_json, output_jld2)

Convert the Python `dense_bosonic_triangle_iPESS_v1` JSON format to dense
TensorKit `B_set` and `T_set` matrices. The output can be passed directly to
`FullUpdate_iPESS_J1_Jchi_up.jl`.
"""
function dense_bosonic_ipess_json_to_jld2(
    input_json::AbstractString,
    output_jld2::AbstractString,
)
    input_json = abspath(input_json)
    output_jld2 = abspath(output_jld2)
    payload = _read_json_dictionary(input_json)
    get(payload, "format", nothing) == DENSE_BOSONIC_IPESS_FORMAT || error(
        "Unsupported or missing JSON format. Expected " *
        "`$DENSE_BOSONIC_IPESS_FORMAT`, got $(get(payload, "format", nothing))."
    )
    haskey(payload, "B_set") || error("JSON state has no `B_set`.")
    haskey(payload, "T_set") || error("JSON state has no `T_set`.")
    B_dictionary = payload["B_set"]
    T_dictionary = payload["T_set"]
    Lx, Ly = _cell_size(B_dictionary, T_dictionary)

    B_set = Matrix{TensorMap}(undef, Lx, Ly)
    T_set = Matrix{TensorMap}(undef, Lx, Ly)
    for cx in 1:Lx, cy in 1:Ly
        key = "$cx,$cy"
        B_entry = B_dictionary[key]
        T_entry = T_dictionary[key]
        _check_signatures(B_entry, B_SIGNATURES, "B_set[$key]")
        _check_signatures(T_entry, T_SIGNATURES, "T_set[$key]")
        B_array = _decode_fortran_array(B_entry, "B_set[$key]", 3)
        T_array = _decode_fortran_array(T_entry, "T_set[$key]", 4)
        B_set[cx, cy] = _array_to_simplex_tensor(B_array)
        T_set[cx, cy] = _array_to_site_tensor(T_array)
    end
    D, physical_dim = _validate_dense_ipess_dimensions(B_set, T_set)

    metadata = (
        source_format=DENSE_BOSONIC_IPESS_FORMAT,
        source_json=input_json,
        array_order="F",
        B_leg_order=("L", "U", "M"),
        T_leg_order=("M", "s", "R", "D"),
        cell_size=(Lx, Ly),
        bond_dimension=D,
        physical_dimension=physical_dim,
    )
    jldsave(output_jld2; B_set=B_set, T_set=T_set, metadata=metadata)
    println("Converted dense bosonic triangular iPESS JSON -> JLD2")
    println("  input  = $input_json")
    println("  output = $output_jld2")
    println("  cell   = $(Lx)x$(Ly), D=$D, physical_dim=$physical_dim, order=F")
    return output_jld2
end


function _load_jld2_ipess_sets(input_jld2::AbstractString)
    data = load(input_jld2)
    if haskey(data, "B_set") && haskey(data, "T_set")
        return data["B_set"], data["T_set"]
    elseif haskey(data, "x")
        state = data["x"]
        Lx, Ly = size(state)
        B_set = Matrix{TensorMap}(undef, Lx, Ly)
        T_set = Matrix{TensorMap}(undef, Lx, Ly)
        for cx in 1:Lx, cy in 1:Ly
            hasproperty(state[cx, cy], :Tm) || error("x[$cx,$cy] has no simplex tensor `Tm`.")
            hasproperty(state[cx, cy], :Bm) || error("x[$cx,$cy] has no site tensor `Bm`.")
            B_set[cx, cy] = state[cx, cy].Tm
            T_set[cx, cy] = state[cx, cy].Bm
        end
        return B_set, T_set
    end
    error("JLD2 state must contain either B_set/T_set or x.")
end


function _encode_fortran_tensor(tensor, signatures)
    dense = Array{ComplexF64}(convert(Array, tensor))
    # `vec` follows Julia column-major order and is therefore exactly the
    # Python `reshape(dense, -1, order="F")` representation.
    flat = vec(dense)
    return Dict(
        "T_real" => real.(flat),
        "T_imag" => imag.(flat),
        "dims" => collect(size(dense)),
        "signatures" => signatures,
    )
end


"""
    dense_bosonic_ipess_jld2_to_json(input_jld2, output_json)

Convert a dense TensorKit triangular iPESS checkpoint back to the Python JSON
format. Both `B_set`/`T_set` checkpoints and checkpoints containing `x` are
accepted.
"""
function dense_bosonic_ipess_jld2_to_json(
    input_jld2::AbstractString,
    output_json::AbstractString,
)
    input_jld2 = abspath(input_jld2)
    output_json = abspath(output_json)
    B_set, T_set = _load_jld2_ipess_sets(input_jld2)
    size(B_set) == size(T_set) || error("B_set and T_set have different cell sizes.")
    Lx, Ly = size(B_set)
    D, physical_dim = _validate_dense_ipess_dimensions(B_set, T_set)

    B_dictionary = Dict{String,Any}()
    T_dictionary = Dict{String,Any}()
    for cx in 1:Lx, cy in 1:Ly
        key = "$cx,$cy"
        B_dictionary[key] = _encode_fortran_tensor(B_set[cx, cy], B_SIGNATURES)
        T_dictionary[key] = _encode_fortran_tensor(T_set[cx, cy], T_SIGNATURES)
    end
    payload = Dict(
        "format" => DENSE_BOSONIC_IPESS_FORMAT,
        "B_set" => B_dictionary,
        "T_set" => T_dictionary,
    )
    open(output_json, "w") do io
        JSON.print(io, payload)
    end
    println("Converted dense bosonic triangular iPESS JLD2 -> JSON")
    println("  input  = $input_jld2")
    println("  output = $output_json")
    println("  cell   = $(Lx)x$(Ly), D=$D, physical_dim=$physical_dim, order=F")
    return output_json
end


function convert_dense_bosonic_ipess_state(
    input_file::AbstractString,
    output_file::AbstractString,
)
    input_extension = lowercase(splitext(input_file)[2])
    output_extension = lowercase(splitext(output_file)[2])
    if input_extension == ".json" && output_extension == ".jld2"
        return dense_bosonic_ipess_json_to_jld2(input_file, output_file)
    elseif input_extension == ".jld2" && output_extension == ".json"
        return dense_bosonic_ipess_jld2_to_json(input_file, output_file)
    end
    error("Expected conversion `.json -> .jld2` or `.jld2 -> .json`; got `$input_file -> $output_file`.")
end


if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error(
        "Usage: julia convert_dense_bosonic_iPESS_json_jld2.jl INPUT.json OUTPUT.jld2\n" *
        "   or: julia convert_dense_bosonic_iPESS_json_jld2.jl INPUT.jld2 OUTPUT.json"
    )
    convert_dense_bosonic_ipess_state(ARGS[1], ARGS[2])
end




#usage:

input_file = "triangle_spin_J1_1_Jchi_0p4_D6_chi80_Lx2_Ly1_D6_to_D8.jld2"

output_file = "triangle_spin_J1_1_Jchi_0p4_D6_chi80_Lx2_Ly1_D6_to_D8.json"

# # Python JSON → Julia JLD2
# convert_dense_bosonic_ipess_state(input_file, output_file)

# # Julia JLD2 → Python JSON
# convert_dense_bosonic_ipess_state(input_file, output_file)