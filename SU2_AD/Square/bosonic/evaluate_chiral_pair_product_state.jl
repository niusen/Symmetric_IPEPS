using TensorKit
using LinearAlgebra: norm
using JLD2
using KrylovKit
using JSON
using Random
using Zygote: @ignore_derivatives

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_chiral_pair_model.jl"))

Random.seed!(1234)

# Change this to a saved single-copy chiral d=2 state.
# The file should contain either `A`, or `x[1,1].T`.
chiral_state_filenm = "Optim_RVB_coefficients_D3_chi_54_-0.9834.jld2"

chi = 100

J1 = 2 * cos(0.06 * pi) * cos(0.14 * pi)
J2 = 2 * cos(0.06 * pi) * sin(0.14 * pi)
Jchi = 2 * sin(0.06 * pi) * 2
parameters = Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)])

ctm_setting = LS_CTMRG_settings()
ctm_setting.CTM_conv_tol = 1e-6
ctm_setting.CTM_ite_nums = 50
ctm_setting.CTM_trun_tol = 1e-8
ctm_setting.svd_lanczos_tol = 1e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = true
ctm_setting.CTM_conv_info = true
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true
ctm_setting.grad_checkpoint = false
dump(ctm_setting)

global multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = ctm_setting.CTM_trun_tol

function load_single_copy_chiral_tensor(filenm)
    data = load(filenm)
    if haskey(data, "A")
        return data["A"]
    elseif haskey(data, "x")
        x = data["x"]
        @assert size(x) == (1, 1) "Only a 1x1 saved `x` cell is supported."
        return x[1, 1].T
    else
        error("Cannot find tensor. Expected saved key `A`, or saved cell `x` with `x[1,1].T`.")
    end
end

function tensor_data_conjugate(A::TensorMap)
    Abar = deepcopy(A)
    Abar.data .= conj.(Abar.data)
    return Abar
end

function tensor_product_chiral_pair(A::TensorMap; normalize_input::Bool=true)
    if normalize_input
        A = A / norm(A)
    end
    Abar = tensor_data_conjugate(A)

    println("single-copy spaces:")
    println("  virtual spaces = " *
            string((space(A, 1), space(A, 2), space(A, 3), space(A, 4))))
    println("  physical space = " * string(space(A, 5)) * ", dim=" * string(dim(space(A, 5))))
    flush(stdout)

    @assert dim(space(A, 5)) == 2 "Input chiral tensor should have a two-dimensional physical space."

    U1 = unitary(fuse(space(A, 1) * space(Abar, 1)), space(A, 1) * space(Abar, 1))
    U2 = unitary(fuse(space(A, 2) * space(Abar, 2)), space(A, 2) * space(Abar, 2))
    U3 = unitary(fuse(space(A, 3) * space(Abar, 3)), space(A, 3) * space(Abar, 3))
    U4 = unitary(fuse(space(A, 4) * space(Abar, 4)), space(A, 4) * space(Abar, 4))
    Up = unitary(fuse(space(A, 5) * space(Abar, 5)), space(A, 5) * space(Abar, 5))

    # The chiral-pair Hamiltonian uses Jchi * (chi_A - chi_B).
    # Put conj(A) in copy A and A in copy B, so the product state has the
    # expected negative chiral energy for a positive-Jchi chiral state.
    @tensor A_product[:] := Abar[-1, -3, -5, -7, -9] *
        A[-2, -4, -6, -8, -10]

    @tensor A_pair[:] := A_product[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] *
        U1[-1, 1, 2] *
        U2[-2, 3, 4] *
        U3[-3, 5, 6] *
        U4[-4, 7, 8] *
        Up[-5, 9, 10]

    A_pair = A_pair / norm(A_pair)

    println("chiral-pair product spaces:")
    println("  virtual spaces = " *
            string((space(A_pair, 1), space(A_pair, 2), space(A_pair, 3), space(A_pair, 4))))
    println("  physical space = " * string(space(A_pair, 5)) * ", dim=" * string(dim(space(A_pair, 5))))
    flush(stdout)

    @assert space(A_pair, 5) == chiral_pair_physical_space()
    return A_pair
end

function evaluate_loaded_chiral_product_state(chiral_state_filenm; chi=chi, parameters=parameters, ctm_setting=ctm_setting)
    A_single = load_single_copy_chiral_tensor(chiral_state_filenm)
    A_pair = tensor_product_chiral_pair(A_single)
    state = Square_iPEPS(A_pair)

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    E, E_triangles, obs, ite_num, ite_err, CTM =
        energy_CTM_chiral_pair_with_observables(state, chi, parameters, ctm_setting, init, [])

    println("chiral-pair product-state energy:")
    println("  E = " * string(E))
    println("  E_triangle = " * string(real.(collect(E_triangles))))
    println("  ctm_ite_num = " * string(ite_num))
    println("  ctm_ite_err = " * string(ite_err))
    print_chiral_pair_observables(obs)
    flush(stdout)

    save_filenm = "chiral_pair_product_from_single_copy_chi_" * string(chi) * ".jld2"
    jldsave(
        save_filenm;
        A=A_pair,
        A_single=A_single,
        energy=E,
        E_triangles=E_triangles,
        observables=obs,
        parameters=parameters,
    )
    println("Saved chiral-pair product state and energy to " * save_filenm)
    flush(stdout)

    return (
        energy=E,
        E_triangles=E_triangles,
        observables=obs,
        ctm_ite_num=ite_num,
        ctm_ite_err=ite_err,
        save_filenm=save_filenm,
    )
end

result = evaluate_loaded_chiral_product_state(chiral_state_filenm; chi=chi)
nothing
