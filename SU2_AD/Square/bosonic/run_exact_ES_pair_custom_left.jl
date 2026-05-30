using LinearAlgebra: I, diag, norm
using TensorKit
using KrylovKit
using JLD2
using MAT
using Random
using Zygote: @ignore_derivatives

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_chiral_pair_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mps_algorithms", "ES_exact_iPEPS_bosonic.jl"))

function include_single_function_definition(filename::AbstractString, fname::AbstractString)
    lines = readlines(filename)
    start = findfirst(line -> occursin("function " * fname * "(", line), lines)
    @assert start !== nothing "Cannot find function `$(fname)` in $(filename)."

    buffer = String[]
    depth = 0
    for line in lines[start:end]
        push!(buffer, line)
        code = strip(split(line, "#"; limit=2)[1])
        depth += length(collect(eachmatch(r"\b(function|if|for|while|let|try|begin|quote|macro)\b", code)))
        depth += occursin(r"^\s*mutable\s+struct\b", code) ? 1 : 0
        depth += occursin(r"^\s*struct\b", code) ? 1 : 0
        if occursin(r"^\s*end\s*$", code)
            depth -= 1
            if depth == 0
                break
            end
        end
    end
    @assert depth == 0 "Function `$(fname)` extraction from $(filename) did not close cleanly."
    include_string(@__MODULE__, join(buffer, "\n"), filename)
    return nothing
end

include_single_function_definition(joinpath(@__DIR__, "ES_pair_custom_left.jl"), "build_custom_left_double_layer")

Random.seed!(555)

init_statenm = "OptimKit_SU2_chiral_pair_D6_chi_54_-1.9594.jld2"
Ly_set = [4, ]
EH_n = 200
krylovdim = 40

function _su2_sector_spin_value(sec)
    if hasproperty(sec, :j)
        return Float64(getproperty(sec, :j))
    end
    if hasproperty(sec, :sectors)
        secs = getproperty(sec, :sectors)
        for s in secs
            if hasproperty(s, :j)
                return Float64(getproperty(s, :j))
            end
        end
    end
    error("Cannot extract SU(2) spin from sector " * string(sec))
end

function _diagonal_block_data(D, sec, cc)
    try
        return block(D, sec)
    catch
        data = getproperty(D, :data)
        if hasproperty(data, :values)
            return data.values[cc]
        end
        return data[cc]
    end
end

function _diagonal_tensor_values_and_spin_labels(D)
    V = space(D, 1)
    vals = ComplexF64[]
    Sectors = Float64[]
    for (vv,dat) in blocks(D)
        vals=vcat(vals,diag(dat))
        Sectors=vcat(Sectors,vv.j*ones(length(diag(dat))))
    end
    return vals, Sectors
end

function _diagonal_tensor_values(D)
    V = space(D, 1)
    vals = ComplexF64[]
    for cc in eachindex(V.dims.keys)
        sec = V.dims.keys[cc]
        append!(vals, ComplexF64.(diag(_diagonal_block_data(D, sec, cc))))
    end
    return vals
end

function exact_es_k_phase_blockwise(ev, Ly::Int)
    ev_translation, ev0 = _translate_es_vectors(ev, Ly)
    if Ly == 4
        @tensor k_matrix[:] := ev_translation'[1, 2, 3, 4, -1] *
            ev0[1, 2, 3, 4, -2]
    elseif Ly == 6
        @tensor k_matrix[:] := ev_translation'[1, 2, 3, 4, 5, 6, -1] *
            ev0[1, 2, 3, 4, 5, 6, -2]
    elseif Ly == 8
        @tensor k_matrix[:] := ev_translation'[1, 2, 3, 4, 5, 6, 7, 8, -1] *
            ev0[1, 2, 3, 4, 5, 6, 7, 8, -2]
    else
        error("exact pair-custom-left ES is implemented for Ly=4, Ly=6, and Ly=8.")
    end
    return _diagonal_tensor_values(k_matrix)
end

function exact_pair_custom_left_dominant_boundaries(AA_left, AA_right, Ly::Int; krylovdim::Int=40)
    vr_init = _random_boundary_tensor(space(AA_right, 1), Ly)
    vl_init = _random_boundary_tensor(space(AA_left, 3), Ly)

    right_fun(x) = exact_transfer_right_action(AA_right, x, Ly)
    left_fun(x) = exact_transfer_left_action(AA_left, x, Ly)

    eur, evr = eigsolve(right_fun, vr_init, 1, :LM, Arnoldi(krylovdim=krylovdim))
    eul, evl = eigsolve(left_fun, vl_init, 1, :LM, Arnoldi(krylovdim=krylovdim))

    ir = findmax(abs.(eur))[2]
    il = findmax(abs.(eul))[2]
    return evr[ir], evl[il], eur[ir], eul[il]
end

function exact_pair_custom_left_ES(A, Ly::Int; EH_n::Int=200, krylovdim::Int=40,
        save_filenm=nothing)
    @assert (Ly in (4, 6, 8)) "exact pair-custom-left ES is implemented for Ly=4, Ly=6, and Ly=8."

    AA_right, U_L_right, _, _, _ = exact_bosonic_double_layer(A)
    AA_left, _, _, U_R_left, _ = build_custom_left_double_layer(A)

    @assert space(AA_right, 1) == space(AA_left, 1) "Left boundary spaces of right/left transfer matrices do not match."
    @assert space(AA_right, 3) == space(AA_left, 3) "Right boundary spaces of right/left transfer matrices do not match."

    println("Exact chiral-pair custom-left cylinder ES without CTMRG")
    println("  Ly = " * string(Ly))
    println("  single-layer virtual spaces = " * string((space(A, 1), space(A, 2), space(A, 3), space(A, 4))))
    println("  right AA spaces = " * string(space(AA_right)))
    println("  left  AA spaces = " * string(space(AA_left)))
    flush(stdout)

    VR, VL, lambda_R, lambda_L = exact_pair_custom_left_dominant_boundaries(
        AA_left, AA_right, Ly; krylovdim=krylovdim,
    )
    rho = exact_boundary_density_matrix(VR, VL, U_L_right, U_R_left, Ly)

    left_inds = Tuple(1:Ly)
    right_inds = Tuple((Ly + 1):(2 * Ly))
    eu, ev = eig(rho, left_inds, right_inds)
    eu, Spin = _diagonal_tensor_values_and_spin_labels(eu)



    if save_filenm === nothing
        save_filenm = "ES_exact_pair_custom_left_Ly$(Ly).mat"
    end
    matwrite(save_filenm, Dict(
        "eu" => eu,
        "Spin" => Spin,
        "Ly" => Ly,
        "lambda_R" => lambda_R,
        "lambda_L" => lambda_L,
        "right_AA_spaces" => string(space(AA_right)),
        "left_AA_spaces" => string(space(AA_left)),
    ); compress=false)

    println("Saved exact pair-custom-left ES to " * save_filenm)
    flush(stdout)
    return (
        eu=eu,
        Spin=Spin,
        lambda_R=lambda_R,
        lambda_L=lambda_L,
    )
end

function _load_pair_state_for_exact_custom_left(statenm::AbstractString; tensor_key::String="A")
    data = load(statenm)
    @assert haskey(data, tensor_key) "State file must contain tensor key `$(tensor_key)`."
    A = data[tensor_key]
    @assert space(A, 5) == chiral_pair_physical_space()' "Input tensor must be a d=4 chiral-pair tensor."
    return A / norm(A)
end

function _exact_pair_custom_left_filename(statenm::AbstractString, Ly::Int)
    tag = splitext(basename(statenm))[1]
    return "ES_exact_pair_custom_left_$(tag)_Ly$(Ly).mat"
end

function run_exact_pair_custom_left_ES(statenm::AbstractString=init_statenm;
        Ly_values=Ly_set,
        EH_n_value::Int=EH_n,
        krylovdim_value::Int=krylovdim,
        tensor_key::String="A")

    A = _load_pair_state_for_exact_custom_left(statenm; tensor_key=tensor_key)

    println("Run exact chiral-pair custom-left ES without CTMRG")
    println("  state file = " * statenm)
    println("  tensor space = " * string(space(A)))
    println("  Ly values = " * string(Ly_values))
    println("  EH_n = " * string(EH_n_value))
    println("  krylovdim = " * string(krylovdim_value))
    flush(stdout)

    results = Dict{Int,Any}()
    for Ly in Ly_values
        save_filenm = _exact_pair_custom_left_filename(statenm, Ly)
        results[Ly] = exact_pair_custom_left_ES(
            A,
            Ly;
            EH_n=EH_n_value,
            krylovdim=krylovdim_value,
            save_filenm=save_filenm,
        )
    end
    return results
end

results = run_exact_pair_custom_left_ES()

nothing
