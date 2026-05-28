function exact_bosonic_double_layer(A)
    U_L = unitary(fuse(space(A, 1)' * space(A, 1)), space(A, 1)' * space(A, 1)) * (1 + 0im)
    U_D = unitary(fuse(space(A, 2)' * space(A, 2)), space(A, 2)' * space(A, 2)) * (1 + 0im)
    U_R = unitary(space(A, 3) * space(A, 3)', fuse(space(A, 3)' * space(A, 3))) * (1 + 0im)
    U_U = unitary(space(A, 4) * space(A, 4)', fuse(space(A, 4)' * space(A, 4))) * (1 + 0im)

    @tensor AA[:] := A'[2, 4, 6, 8, 1] * A[3, 5, 7, 9, 1] *
        U_L[-1, 2, 3] * U_D[-2, 4, 5] * U_R[6, 7, -3] * U_U[8, 9, -4]
    return AA, U_L, U_D, U_R, U_U
end

function _repeated_space(V, N::Int)
    Vout = V
    for _ in 2:N
        Vout = Vout * V
    end
    return Vout
end

function _random_boundary_tensor(V, Ly::Int)
    cod = _repeated_space(V, Ly)
    return TensorMap(randn, cod, one(cod)) + im * TensorMap(randn, cod, one(cod))
end

function exact_transfer_right_action(AA, vr0, Ly::Int)
    if Ly == 4
        @tensor vr[:] := AA[-1, 2, 1, 8] * AA[-2, 4, 3, 2] *
            AA[-3, 6, 5, 4] * AA[-4, 8, 7, 6] * vr0[1, 3, 5, 7]
    elseif Ly == 6
        @tensor vr[:] := AA[-1, 2, 1, 12] * AA[-2, 4, 3, 2] *
            AA[-3, 6, 5, 4] * AA[-4, 8, 7, 6] *
            AA[-5, 10, 9, 8] * AA[-6, 12, 11, 10] *
            vr0[1, 3, 5, 7, 9, 11]
    elseif Ly == 8
        @tensor vr[:] := AA[-1, 2, 1, 16] * AA[-2, 4, 3, 2] *
            AA[-3, 6, 5, 4] * AA[-4, 8, 7, 6] *
            AA[-5, 10, 9, 8] * AA[-6, 12, 11, 10] *
            AA[-7, 14, 13, 12] * AA[-8, 16, 15, 14] *
            vr0[1, 3, 5, 7, 9, 11, 13, 15]
    else
        error("exact bosonic iPEPS ES is implemented for Ly=4, Ly=6, and Ly=8.")
    end
    return vr
end

function exact_transfer_left_action(AA, vl0, Ly::Int)
    if Ly == 4
        @tensor vl[:] := AA[1, 2, -1, 8] * AA[3, 4, -2, 2] *
            AA[5, 6, -3, 4] * AA[7, 8, -4, 6] * vl0[1, 3, 5, 7]
    elseif Ly == 6
        @tensor vl[:] := AA[1, 2, -1, 12] * AA[3, 4, -2, 2] *
            AA[5, 6, -3, 4] * AA[7, 8, -4, 6] *
            AA[9, 10, -5, 8] * AA[11, 12, -6, 10] *
            vl0[1, 3, 5, 7, 9, 11]
    elseif Ly == 8
        @tensor vl[:] := AA[1, 2, -1, 16] * AA[3, 4, -2, 2] *
            AA[5, 6, -3, 4] * AA[7, 8, -4, 6] *
            AA[9, 10, -5, 8] * AA[11, 12, -6, 10] *
            AA[13, 14, -7, 12] * AA[15, 16, -8, 14] *
            vl0[1, 3, 5, 7, 9, 11, 13, 15]
    else
        error("exact bosonic iPEPS ES is implemented for Ly=4, Ly=6, and Ly=8.")
    end
    return vl
end

function exact_dominant_boundaries(AA, Ly::Int; krylovdim::Int=40)
    vr_init = _random_boundary_tensor(space(AA, 1), Ly)
    vl_init = _random_boundary_tensor(space(AA, 3), Ly)

    right_fun(x) = exact_transfer_right_action(AA, x, Ly)
    left_fun(x) = exact_transfer_left_action(AA, x, Ly)

    eur, evr = eigsolve(right_fun, vr_init, 1, :LM, Arnoldi(krylovdim=krylovdim))
    eul, evl = eigsolve(left_fun, vl_init, 1, :LM, Arnoldi(krylovdim=krylovdim))

    ir = findmax(abs.(eur))[2]
    il = findmax(abs.(eul))[2]
    return evr[ir], evl[il], eur[ir], eul[il]
end

function split_exact_boundaries(VR, VL, U_L, U_R, Ly::Int)
    if Ly == 4
        @tensor VLs[:] := VL[1, 2, 3, 4] *
            U_R'[1, -1, -5] * U_R'[2, -2, -6] *
            U_R'[3, -3, -7] * U_R'[4, -4, -8]
        @tensor VRs[:] := VR[1, 2, 3, 4] *
            U_L'[-5, -1, 1] * U_L'[-6, -2, 2] *
            U_L'[-7, -3, 3] * U_L'[-8, -4, 4]
    elseif Ly == 6
        @tensor VLs[:] := VL[1, 2, 3, 4, 5, 6] *
            U_R'[1, -1, -7] * U_R'[2, -2, -8] *
            U_R'[3, -3, -9] * U_R'[4, -4, -10] *
            U_R'[5, -5, -11] * U_R'[6, -6, -12]
        @tensor VRs[:] := VR[1, 2, 3, 4, 5, 6] *
            U_L'[-7, -1, 1] * U_L'[-8, -2, 2] *
            U_L'[-9, -3, 3] * U_L'[-10, -4, 4] *
            U_L'[-11, -5, 5] * U_L'[-12, -6, 6]
    elseif Ly == 8
        @tensor VLs[:] := VL[1, 2, 3, 4, 5, 6, 7, 8] *
            U_R'[1, -1, -9] * U_R'[2, -2, -10] *
            U_R'[3, -3, -11] * U_R'[4, -4, -12] *
            U_R'[5, -5, -13] * U_R'[6, -6, -14] *
            U_R'[7, -7, -15] * U_R'[8, -8, -16]
        @tensor VRs[:] := VR[1, 2, 3, 4, 5, 6, 7, 8] *
            U_L'[-9, -1, 1] * U_L'[-10, -2, 2] *
            U_L'[-11, -3, 3] * U_L'[-12, -4, 4] *
            U_L'[-13, -5, 5] * U_L'[-14, -6, 6] *
            U_L'[-15, -7, 7] * U_L'[-16, -8, 8]
    else
        error("exact bosonic iPEPS ES is implemented for Ly=4, Ly=6, and Ly=8.")
    end
    return VRs, VLs
end

function exact_boundary_density_matrix(VR, VL, U_L, U_R, Ly::Int)
    VRs, VLs = split_exact_boundaries(VR, VL, U_L, U_R, Ly)
    if Ly == 4
        @tensor rho[:] := VLs[-1, -2, -3, -4, 1, 2, 3, 4] *
            VRs[1, 2, 3, 4, -5, -6, -7, -8]
    elseif Ly == 6
        @tensor rho[:] := VLs[-1, -2, -3, -4, -5, -6, 1, 2, 3, 4, 5, 6] *
            VRs[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12]
    elseif Ly == 8
        @tensor rho[:] := VLs[-1, -2, -3, -4, -5, -6, -7, -8, 1, 2, 3, 4, 5, 6, 7, 8] *
            VRs[1, 2, 3, 4, 5, 6, 7, 8, -9, -10, -11, -12, -13, -14, -15, -16]
    else
        error("exact bosonic iPEPS ES is implemented for Ly=4, Ly=6, and Ly=8.")
    end
    return rho
end

function _translate_es_vectors(ev, Ly::Int)
    if Ly == 4
        ev = permute(ev, (1, 2, 3, 4, 5,), ())
        return permute(ev, (2, 3, 4, 1, 5,), ()), ev
    elseif Ly == 6
        ev = permute(ev, (1, 2, 3, 4, 5, 6, 7,), ())
        return permute(ev, (2, 3, 4, 5, 6, 1, 7,), ()), ev
    elseif Ly == 8
        ev = permute(ev, (1, 2, 3, 4, 5, 6, 7, 8, 9,), ())
        return permute(ev, (2, 3, 4, 5, 6, 7, 8, 1, 9,), ()), ev
    else
        error("exact bosonic iPEPS ES is implemented for Ly=4, Ly=6, and Ly=8.")
    end
end

function exact_es_k_phase(ev, Ly::Int)
    ev_translation, ev0 = _translate_es_vectors(ev, Ly)
    if Ly == 4
        @tensor k_matrix[:] := ev_translation'[1, 2, 3, 4, -1] * ev0[1, 2, 3, 4, -2]
    elseif Ly == 6
        @tensor k_matrix[:] := ev_translation'[1, 2, 3, 4, 5, 6, -1] *
            ev0[1, 2, 3, 4, 5, 6, -2]
    elseif Ly == 8
        @tensor k_matrix[:] := ev_translation'[1, 2, 3, 4, 5, 6, 7, 8, -1] *
            ev0[1, 2, 3, 4, 5, 6, 7, 8, -2]
    else
        error("exact bosonic iPEPS ES is implemented for Ly=4, Ly=6, and Ly=8.")
    end
    return diag(convert(Array, k_matrix))
end

function exact_bosonic_iPEPS_ES(A, Ly::Int; EH_n::Int=200, krylovdim::Int=40,
        save_filenm=nothing)
    @assert Ly == 4 || Ly == 6 || Ly == 8 "exact bosonic iPEPS ES is implemented for Ly=4, Ly=6, and Ly=8."

    AA, U_L, U_D, U_R, U_U = exact_bosonic_double_layer(A)
    println("Exact iPEPS cylinder ES without CTMRG")
    println("  Ly = " * string(Ly))
    println("  single-layer virtual dim = " * string(dim(space(A, 1))))
    println("  boundary Hilbert dim = " * string(dim(space(A, 1))^Ly))
    flush(stdout)

    VR, VL, lambda_R, lambda_L = exact_dominant_boundaries(AA, Ly; krylovdim=krylovdim)
    rho = exact_boundary_density_matrix(VR, VL, U_L, U_R, Ly)

    left_inds = Tuple(1:Ly)
    right_inds = Tuple((Ly + 1):(2 * Ly))
    eu, ev = eig(rho, left_inds, right_inds)
    eu = diag(convert(Array, eu))
    eu = ComplexF64.(eu)
    eu = eu / sum(eu)
    entanglement_spectrum = -log.(abs.(eu))
    k_phase = exact_es_k_phase(ev, Ly)

    order = sortperm(abs.(eu); rev=true)
    keep = order[1:min(EH_n, length(order))]
    eu = eu[keep]
    entanglement_spectrum = entanglement_spectrum[keep]
    k_phase = k_phase[keep]

    if save_filenm === nothing
        save_filenm = "ES_exact_iPEPS_Ly$(Ly).mat"
    end
    matwrite(save_filenm, Dict(
        "eu" => eu,
        "entanglement_spectrum" => entanglement_spectrum,
        "k_phase" => k_phase,
        "Ly" => Ly,
        "lambda_R" => lambda_R,
        "lambda_L" => lambda_L,
        "boundary_dim" => dim(space(A, 1))^Ly,
    ); compress=false)

    println("Saved exact iPEPS ES to " * save_filenm)
    flush(stdout)
    return (
        eu=eu,
        entanglement_spectrum=entanglement_spectrum,
        k_phase=k_phase,
        rho=rho,
        lambda_R=lambda_R,
        lambda_L=lambda_L,
    )
end
