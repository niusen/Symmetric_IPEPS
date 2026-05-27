using TensorKit
using LinearAlgebra: norm
using JLD2
using Zygote: @ignore_derivatives

function chiral_pair_copy_space()
    return SU2Space(1 / 2 => 1)
end

function chiral_pair_physical_space()
    Vcopy = chiral_pair_copy_space()
    return fuse(Vcopy * Vcopy)
end

const CHIRAL_PAIR_FUSE_UNITARY_CACHE = Ref{Any}(nothing)
const CHIRAL_PAIR_VIRTUAL_ISOMETRY_CACHE = Dict{Tuple{Any, Any}, Any}()

function chiral_pair_fuse_unitary()
    if CHIRAL_PAIR_FUSE_UNITARY_CACHE[] === nothing
        Vcopy = chiral_pair_copy_space()
        CHIRAL_PAIR_FUSE_UNITARY_CACHE[] = unitary(fuse(Vcopy * Vcopy), Vcopy * Vcopy)
    end
    return CHIRAL_PAIR_FUSE_UNITARY_CACHE[]
end

function chiral_pair_virtual_isometry(Vsmall, Vbig)
    @assert dim(Vbig) >= dim(Vsmall) "Cannot embed a larger virtual space into a smaller one."
    key = (Vsmall, Vbig)
    if !haskey(CHIRAL_PAIR_VIRTUAL_ISOMETRY_CACHE, key)
        W = isometry(Vbig, Vsmall)
        @assert norm(W' * W - unitary(Vsmall, Vsmall)) < 1e-12 "Virtual embedding isometry does not satisfy W' * W = I."
        CHIRAL_PAIR_VIRTUAL_ISOMETRY_CACHE[key] = W
    end
    return CHIRAL_PAIR_VIRTUAL_ISOMETRY_CACHE[key]
end

function embed_chiral_pair_tensor_virtual_space(A, Vbig)
    Vsmall = space(A, 1)
    if Vsmall == Vbig
        return A
    end
    @assert space(A, 2) == Vsmall "Only C4-like tensors with equal first two virtual spaces are supported."
    @assert space(A, 3) == Vsmall' "Expected third virtual leg to be the dual of the first leg."
    @assert space(A, 4) == Vsmall' "Expected fourth virtual leg to be the dual of the first leg."

    W = chiral_pair_virtual_isometry(Vsmall, Vbig)
    @tensor A_big[:] := W[-1, 1] * W[-2, 2] * W'[3, -3] * W'[4, -4] *
        A[1, 2, 3, 4, -5]
    return A_big
end

function initial_SU2_chiral_pair_state(
    Vspace,
    init_statenm="nothing",
    init_noise=0;
    init_complex_tensor=true,
    init_C4_symetry=false,
)
    Vp_pair = chiral_pair_physical_space()

    if init_statenm == "nothing"
        println("Random initial chiral-pair state")
        flush(stdout)

        if init_complex_tensor
            A = TensorMap(randn, Vspace * Vspace * Vspace * Vspace, Vp_pair) +
                im * TensorMap(randn, Vspace * Vspace * Vspace * Vspace, Vp_pair)
        else
            A = TensorMap(randn, Vspace * Vspace * Vspace * Vspace, Vp_pair)
        end

        if init_C4_symetry
            A = permute(A, (1, 2, 3, 4, 5,)) +
                permute(A, (2, 3, 4, 1, 5,)) +
                permute(A, (3, 4, 1, 2, 5,)) +
                permute(A, (4, 1, 2, 3, 5,))
            A = A / norm(A)
        else
            A = permute(A, (1, 2, 3, 4, 5,))
            A = A / norm(A)
        end

        U = unitary(Vspace', Vspace)
        @tensor A[:] := A[-1, -2, 1, 2, -5] * U[-3, 1] * U[-4, 2]
        return Square_iPEPS(A)
    end

    println("load chiral-pair state: " * init_statenm)
    flush(stdout)
    data = load(init_statenm)
    @assert haskey(data, "A") "Initial state file must contain tensor `A`."
    A = data["A"]

    @assert space(A, 5) == Vp_pair' "Loaded physical space is not the chiral-pair physical leg Rep[SU2](0=>1, 1=>1)'."
    if space(A, 1) != Vspace
        println("Embed loaded chiral-pair virtual space from " * string(space(A, 1)) *
                " to " * string(Vspace))
        flush(stdout)
        A = embed_chiral_pair_tensor_virtual_space(A, Vspace)
    end
    @assert space(A, 1) == Vspace "Loaded virtual space does not match requested Vspace after embedding."

    if init_noise != 0
        if init_complex_tensor
            A_noise = TensorMap(randn, codomain(A), domain(A)) +
                im * TensorMap(randn, codomain(A), domain(A))
        else
            A_noise = TensorMap(randn, codomain(A), domain(A))
        end
        A = A + A_noise * init_noise * norm(A) / norm(A_noise)
    end

    if init_C4_symetry
        U = unitary(space(A, 3)', space(A, 3))
        @tensor A[:] := A[-1, -2, 1, 2, -5] * U[-3, 1] * U[-4, 2]
        A = permute(A, (1, 2, 3, 4, 5,)) +
            permute(A, (2, 3, 4, 1, 5,)) +
            permute(A, (3, 4, 1, 2, 5,)) +
            permute(A, (4, 1, 2, 3, 5,))
        A = A / norm(A)
        U = unitary(space(A, 3)', space(A, 3))
        @tensor A[:] := A[-1, -2, 1, 2, -5] * U[-3, 1] * U[-4, 2]
    end

    return Square_iPEPS(A)
end

function chiral_pair_ctmrg(A, chi, init, init_CTM, ctm_setting)
    out = CTMRG(A, chi, init, init_CTM, ctm_setting)
    if length(out) == 8
        return out
    elseif length(out) == 6
        CTM, AA, U_L, U_D, U_R, U_U = out
        return CTM, AA, U_L, U_D, U_R, U_U, missing, missing
    else
        error("Unexpected CTMRG return length: " * string(length(out)))
    end
end

function _chiral_pair_spin_ops()
    _, H123chiral, H12, H31, H23 = Hamiltonians(chiral_pair_copy_space())
    return H12, H31, H23, H123chiral
end

function _chiral_pair_reduce_copy_rho(rho_raw, U_s_s, copy::Symbol)
    @assert copy in (:A, :B)

    U_pair = @ignore_derivatives chiral_pair_fuse_unitary()

    @tensor rho_pair[:] := rho_raw[1, 2, 3] *
        U_s_s[-1, -4, 1] *
        U_s_s[-2, -5, 2] *
        U_s_s[-3, -6, 3]

    if copy == :A
        @tensor rho_copy[:] := rho_pair[1, 2, 3, 4, 5, 6] *
            U_pair'[-1, 7, 1] *
            U_pair'[-2, 8, 2] *
            U_pair'[-3, 9, 3] *
            U_pair[4, -4, 7] *
            U_pair[5, -5, 8] *
            U_pair[6, -6, 9]
    else
        @tensor rho_copy[:] := rho_pair[1, 2, 3, 4, 5, 6] *
            U_pair'[7, -1, 1] *
            U_pair'[8, -2, 2] *
            U_pair'[9, -3, 3] *
            U_pair[4, 7, -4] *
            U_pair[5, 8, -5] *
            U_pair[6, 9, -6]
    end

    return rho_copy
end

function _chiral_pair_triangle_expectations(rho_raw, U_s_s)
    H12, H31, H23, H123chiral = @ignore_derivatives _chiral_pair_spin_ops()

    rho_A = _chiral_pair_reduce_copy_rho(rho_raw, U_s_s, :A)
    norm_A = @tensor rho_A[1, 2, 3, 1, 2, 3]
    e12_A = @tensor rho_A[1, 2, 3, 4, 5, 6] * H12[1, 2, 3, 4, 5, 6]
    e31_A = @tensor rho_A[1, 2, 3, 4, 5, 6] * H31[1, 2, 3, 4, 5, 6]
    e23_A = @tensor rho_A[1, 2, 3, 4, 5, 6] * H23[1, 2, 3, 4, 5, 6]
    chi_A = @tensor rho_A[1, 2, 3, 4, 5, 6] * H123chiral[1, 2, 3, 4, 5, 6]

    rho_B = _chiral_pair_reduce_copy_rho(rho_raw, U_s_s, :B)
    norm_B = @tensor rho_B[1, 2, 3, 1, 2, 3]
    e12_B = @tensor rho_B[1, 2, 3, 4, 5, 6] * H12[1, 2, 3, 4, 5, 6]
    e31_B = @tensor rho_B[1, 2, 3, 4, 5, 6] * H31[1, 2, 3, 4, 5, 6]
    e23_B = @tensor rho_B[1, 2, 3, 4, 5, 6] * H23[1, 2, 3, 4, 5, 6]
    chi_B = @tensor rho_B[1, 2, 3, 4, 5, 6] * H123chiral[1, 2, 3, 4, 5, 6]

    return (
        e12_A=e12_A / norm_A,
        e31_A=e31_A / norm_A,
        e23_A=e23_A / norm_A,
        chi_A=chi_A / norm_A,
        e12_B=e12_B / norm_B,
        e31_B=e31_B / norm_B,
        e23_B=e23_B / norm_B,
        chi_B=chi_B / norm_B,
    )
end

function _chiral_pair_weight_triangle_terms(t, parameters)
    J1 = parameters["J1"]
    J2 = parameters["J2"]
    Jchi = parameters["Jchi"]

    heisenberg_nn = J1 / 4 * (t.e12_A + t.e12_B + t.e31_A + t.e31_B)
    heisenberg_nnn = J2 / 2 * (t.e23_A + t.e23_B)
    chirality = Jchi * (t.chi_A - t.chi_B)

    return heisenberg_nn + heisenberg_nnn + chirality
end

function evaluate_chiral_pair_triangle(
    A::TensorMap,
    AA,
    U_L,
    U_D,
    U_R,
    U_U,
    CTM,
    ctm_setting,
    parameters,
)
    AA_open, U_s_s = build_double_layer_open(A)

    rho_LU_RU_LD = ob_LU_RU_LD(CTM, AA, AA_open, AA_open, AA_open)
    t1 = _chiral_pair_triangle_expectations(rho_LU_RU_LD, U_s_s)

    rho_LD_RU_RD = ob_LD_RU_RD(CTM, AA, AA_open, AA_open, AA_open)
    rho_LD_RU_RD = permute(rho_LD_RU_RD, (3, 1, 2,))
    t2 = _chiral_pair_triangle_expectations(rho_LD_RU_RD, U_s_s)

    rho_LU_LD_RD = ob_LU_LD_RD(CTM, AA, AA_open, AA_open, AA_open)
    rho_LU_LD_RD = permute(rho_LU_LD_RD, (2, 1, 3,))
    t3 = _chiral_pair_triangle_expectations(rho_LU_LD_RD, U_s_s)

    rho_LU_RU_RD = ob_LU_RU_RD(CTM, AA, AA_open, AA_open, AA_open)
    rho_LU_RU_RD = permute(rho_LU_RU_RD, (2, 3, 1,))
    t4 = _chiral_pair_triangle_expectations(rho_LU_RU_RD, U_s_s)

    triangles = (t1, t2, t3, t4)
    E_triangles = map(t -> _chiral_pair_weight_triangle_terms(t, parameters), triangles)
    E = sum(E_triangles)

    return E, E_triangles, triangles
end

function chiral_pair_observables(triangles, parameters)
    e12_A = real.([t.e12_A for t in triangles])
    e31_A = real.([t.e31_A for t in triangles])
    e23_A = real.([t.e23_A for t in triangles])
    chi_A = real.([t.chi_A for t in triangles])

    e12_B = real.([t.e12_B for t in triangles])
    e31_B = real.([t.e31_B for t in triangles])
    e23_B = real.([t.e23_B for t in triangles])
    chi_B = real.([t.chi_B for t in triangles])

    nn_A = sum(e12_A .+ e31_A) / 4
    nn_B = sum(e12_B .+ e31_B) / 4
    nnn_A = sum(e23_A) / 2
    nnn_B = sum(e23_B) / 2
    chirality_A = sum(chi_A)
    chirality_B = sum(chi_B)

    J1 = parameters["J1"]
    J2 = parameters["J2"]
    Jchi = parameters["Jchi"]

    return (
        nn_A=nn_A,
        nn_B=nn_B,
        nnn_A=nnn_A,
        nnn_B=nnn_B,
        chirality_A=chirality_A,
        chirality_B=chirality_B,
        e12_A=e12_A,
        e31_A=e31_A,
        e23_A=e23_A,
        chi_A=chi_A,
        e12_B=e12_B,
        e31_B=e31_B,
        e23_B=e23_B,
        chi_B=chi_B,
        weighted_nn=J1 * (nn_A + nn_B),
        weighted_nnn=J2 * (nnn_A + nnn_B),
        weighted_chirality=Jchi * (chirality_A - chirality_B),
    )
end

function print_chiral_pair_observables(obs)
    total = obs.weighted_nn + obs.weighted_nnn + obs.weighted_chirality
    println("  chiral-pair observables:")
    println("    <NN Heisenberg> A/B   = " * string((obs.nn_A, obs.nn_B)) *
            ", contribution=" * string(obs.weighted_nn))
    println("    <NNN Heisenberg> A/B  = " * string((obs.nnn_A, obs.nnn_B)) *
            ", contribution=" * string(obs.weighted_nnn))
    println("    <chirality> A/B       = " * string((obs.chirality_A, obs.chirality_B)) *
            ", Jchi*(A-B)=" * string(obs.weighted_chirality))
    println("    term contribution sum = " * string(total))
    flush(stdout)
    return total
end

function cost_fun(x)
    global chi, parameters, grad_ctm_setting
    A = x.T
    A = A / norm(A)

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err =
        chiral_pair_ctmrg(A, chi, init, [], grad_ctm_setting)

    E, E_triangles, triangles = evaluate_chiral_pair_triangle(
        A,
        AA,
        U_L,
        U_D,
        U_R,
        U_U,
        CTM,
        grad_ctm_setting,
        parameters,
    )
    E = real(E)

    println("E0= " * string(E) *
            ", E_triangle= " * string(real.(collect(E_triangles))) *
            ", ctm_ite_num= " * string(ite_num) *
            ", ctm_ite_err= " * string(ite_err))
    flush(stdout)

    global E_tem, CTM_tem
    CTM_tem = deepcopy(CTM)
    E_tem = deepcopy(E)
    return E
end

function energy_CTM(x, chi, parameters, ctm_setting, energy_setting, init, init_CTM)
    A = x.T
    A = A / norm(A)

    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err =
        chiral_pair_ctmrg(A, chi, init, init_CTM, ctm_setting)

    E, E_triangles, triangles = evaluate_chiral_pair_triangle(
        A,
        AA,
        U_L,
        U_D,
        U_R,
        U_U,
        CTM,
        ctm_setting,
        parameters,
    )
    E = real(E)

    return (
        E,
        E_triangles[1],
        E_triangles[2],
        E_triangles[3],
        E_triangles[4],
        ite_num,
        ite_err,
        CTM,
    )
end

function energy_CTM_chiral_pair_with_observables(x, chi, parameters, ctm_setting, init, init_CTM)
    A = x.T
    A = A / norm(A)

    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err =
        chiral_pair_ctmrg(A, chi, init, init_CTM, ctm_setting)

    E, E_triangles, triangles = evaluate_chiral_pair_triangle(
        A,
        AA,
        U_L,
        U_D,
        U_R,
        U_U,
        CTM,
        ctm_setting,
        parameters,
    )
    obs = chiral_pair_observables(triangles, parameters)

    return real(E), E_triangles, obs, ite_num, ite_err, CTM
end
