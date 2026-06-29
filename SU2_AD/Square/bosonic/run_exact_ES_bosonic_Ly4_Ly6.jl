using LinearAlgebra: norm, diag
using TensorKit
using KrylovKit
using JLD2
using JSON
using MAT
using Random

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "src", "mps_algorithms", "ES_exact_iPEPS_bosonic.jl"))

Random.seed!(555)

init_statenm = "Optim_RVB_coefficients_D3_chi_54_-0.9834.jld2"
Ly_set = [4, 6, 8]
EH_n = 200
krylovdim = 40
enable_juraj_json_reader = false

function _maybe_include_juraj_reader(statenm::AbstractString)
    if enable_juraj_json_reader || endswith(lowercase(statenm), ".json")
        reader = joinpath(@__DIR__, "read_juraj_ipeps_tensor.jl")
        @assert isfile(reader) "Juraj json reader is requested but not found: $(reader)"
        include(reader)
    end
    return nothing
end

function _load_square_ipeps_for_exact_es(statenm::AbstractString; tensor_key::String="A")
    _maybe_include_juraj_reader(statenm)
    if endswith(lowercase(statenm), ".json")
        return read_juraj_ipeps_tensormap(statenm)
    end

    data = load(statenm)
    @assert haskey(data, tensor_key) "State file must contain tensor key `$(tensor_key)`."
    return data[tensor_key]
end

function _exact_es_filename(statenm::AbstractString, Ly::Int)
    tag = splitext(basename(statenm))[1]
    return "ES_exact_iPEPS_$(tag)_Ly$(Ly).mat"
end

function run_exact_bosonic_iPEPS_ES(statenm::AbstractString=init_statenm;
        Ly_values=Ly_set,
        EH_n_value::Int=EH_n,
        krylovdim_value::Int=krylovdim,
        tensor_key::String="A")

    A = _load_square_ipeps_for_exact_es(statenm; tensor_key=tensor_key)
    A = A / norm(A)

    println("Run exact cylinder iPEPS ES without CTMRG")
    println("  state file      = " * statenm)
    println("  tensor space    = " * string(space(A)))
    println("  Ly values       = " * string(Ly_values))
    println("  EH_n            = " * string(EH_n_value))
    println("  krylovdim       = " * string(krylovdim_value))
    flush(stdout)

    results = Dict{Int,Any}()
    for Ly in Ly_values
        save_filenm = _exact_es_filename(statenm, Ly)
        results[Ly] = exact_bosonic_iPEPS_ES(
            A,
            Ly;
            EH_n=EH_n_value,
            krylovdim=krylovdim_value,
            save_filenm=save_filenm,
        )
    end
    return results
end

results = run_exact_bosonic_iPEPS_ES()

nothing
