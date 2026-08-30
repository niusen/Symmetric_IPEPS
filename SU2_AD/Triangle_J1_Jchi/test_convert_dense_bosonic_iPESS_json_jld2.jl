using Test
using JSON
using JLD2

include("convert_dense_bosonic_iPESS_json_jld2.jl")


function test_array(dims, offset)
    values = [
        complex(offset + index / 10, -offset - index / 100)
        for index in 1:prod(dims)
    ]
    return reshape(values, dims)
end


function python_entry(array, signatures)
    flat = vec(array)
    return Dict(
        "T_real" => real.(flat),
        "T_imag" => imag.(flat),
        "dims" => collect(size(array)),
        "signatures" => signatures,
    )
end


@testset "dense bosonic iPESS JSON/JLD2 conversion" begin
    mktempdir() do directory
        input_json = joinpath(directory, "python_state.json")
        intermediate_jld2 = joinpath(directory, "julia_state.jld2")
        output_json = joinpath(directory, "python_roundtrip.json")

        B_arrays = Dict{String,Array{ComplexF64,3}}()
        T_arrays = Dict{String,Array{ComplexF64,4}}()
        B_dictionary = Dict{String,Any}()
        T_dictionary = Dict{String,Any}()
        for cx in 1:2, cy in 1:2
            key = "$cx,$cy"
            offset = 10cx + cy
            B_arrays[key] = test_array((2, 2, 2), offset)
            T_arrays[key] = test_array((2, 2, 2, 2), 100 + offset)
            B_dictionary[key] = python_entry(B_arrays[key], B_SIGNATURES)
            T_dictionary[key] = python_entry(T_arrays[key], T_SIGNATURES)
        end
        payload = Dict(
            "format" => DENSE_BOSONIC_IPESS_FORMAT,
            "B_set" => B_dictionary,
            "T_set" => T_dictionary,
        )
        open(input_json, "w") do io
            JSON.print(io, payload)
        end

        dense_bosonic_ipess_json_to_jld2(input_json, intermediate_jld2)
        data = load(intermediate_jld2)
        @test haskey(data, "B_set")
        @test haskey(data, "T_set")
        @test size(data["B_set"]) == (2, 2)
        @test size(data["T_set"]) == (2, 2)
        for cx in 1:2, cy in 1:2
            key = "$cx,$cy"
            @test convert(Array, data["B_set"][cx, cy]) == B_arrays[key]
            @test convert(Array, data["T_set"][cx, cy]) == T_arrays[key]
            # The simplex M domain and site M codomain must be directly
            # contractible after conversion.
            @test norm(data["B_set"][cx, cy] * data["T_set"][cx, cy]) > 0
        end

        dense_bosonic_ipess_jld2_to_json(intermediate_jld2, output_json)
        roundtrip = _read_json_dictionary(output_json)
        @test roundtrip["format"] == DENSE_BOSONIC_IPESS_FORMAT
        for cx in 1:2, cy in 1:2
            key = "$cx,$cy"
            for (set_name, expected_array, expected_signatures) in (
                ("B_set", B_arrays[key], B_SIGNATURES),
                ("T_set", T_arrays[key], T_SIGNATURES),
            )
                entry = roundtrip[set_name][key]
                @test Int.(entry["dims"]) == collect(size(expected_array))
                @test Int.(entry["signatures"]) == expected_signatures
                decoded = _decode_fortran_array(entry, "$set_name[$key]", ndims(expected_array))
                @test decoded == expected_array
            end
        end
    end
end
