"""
Matrix-free CTMRG entanglement-spectrum routines for dense bosonic tensors.

This is the dense counterpart of `ES_CTM_algorithms_SU2.jl`.  It deliberately
has its own function names and file so that the established symmetry-resolved
code is unchanged.  There are no SU(2) sectors or vison sectors here; the
transfer operator acts directly on a dense rank-`N` boundary tensor.
"""

function ES_CTMRG_prepare_dense_matrixfree(CTM, U_L, U_R; T_scale=1)
    Tleft = T_scale * CTM.Tset.T4 / norm(CTM.Tset.T4)
    Tright = T_scale * CTM.Tset.T2 / norm(CTM.Tset.T2)

    @tensor O1[:] := Tleft[-3, 1, -1] * U_L[1, -2, -4]
    @tensor O2[:] := Tright[-1, 1, -3] * U_R[-4, -2, 1]
    @tensor OO[:] := O1[-2, -3, -5, 1] * O2[-1, 1, -4, -6]

    U_fuse_chichi = unitary(
        fuse(space(OO, 1) * space(OO, 2)),
        space(OO, 1) * space(OO, 2),
    )
    @tensor OO[:] := U_fuse_chichi[-1, 1, 2] *
        OO[1, 2, -2, 3, 4, -4] * U_fuse_chichi'[3, 4, -3]
    return OO
end

function _dense_mf_repeated_space(V, N::Int)
    N >= 1 || error("N must be positive.")
    Vout = V
    for _ in 2:N
        Vout = Vout * V
    end
    return Vout
end

function dense_mf_initial_vector(OO, N::Int)
    V = _dense_mf_repeated_space(space(OO, 2)', N)
    sz = ntuple(cc -> dim(V[cc]), length(V))
    data = randn(Float64, sz...) + im * randn(Float64, sz...)
    return Tensor(data, V)
end

function dense_mf_translate(v, N::Int)
    N in (4, 5, 6, 8) || error("Dense matrix-free ES supports N=4,5,6,8.")
    order = Tuple(vcat(collect(2:N), 1))
    return permute(v, order, ())
end

function dense_mf_k_projection(v_unprojected, N::Int, kn::Int)
    0 <= kn < N || error("Momentum index kn must satisfy 0 <= kn < N.")
    vnorm = dot(v_unprojected, v_unprojected)
    v_work = deepcopy(v_unprojected)
    v_projected = deepcopy(v_unprojected)
    for cc in 1:(N - 1)
        v_work = dense_mf_translate(v_work, N)
        v_projected += exp(-im * (2 * pi * kn / N) * cc) * v_work
    end
    nrm = sqrt(abs(dot(v_projected, v_projected)))
    nrm > 1e-12 || return v_projected
    return v_projected / nrm * sqrt(abs(vnorm))
end

function dense_mf_calculate_k(ev, N::Int)
    ks = Vector{ComplexF64}(undef, length(ev))
    for cc in eachindex(ev)
        v = ev[cc]
        vp = dense_mf_translate(v, N)
        phase = dot(vp, v) / dot(v, v)
        # Keep the convention used by the original SU(2) ES implementation.
        ks[cc] = N == 8 ? phase' : phase
    end
    return ks
end

function CTM_T_action_dense_matrixfree(OO, v0, N::Int; kn=nothing)
    if N == 4
        @tensor v_new[:] := OO[8, 1, 2, -1] * OO[2, 3, 4, -2] *
            OO[4, 5, 6, -3] * OO[6, 7, 8, -4] * v0[1, 3, 5, 7]
    elseif N == 5
        @tensor v_new[:] := OO[10, 1, 2, -1] * OO[2, 3, 4, -2] *
            OO[4, 5, 6, -3] * OO[6, 7, 8, -4] *
            OO[8, 9, 10, -5] * v0[1, 3, 5, 7, 9]
    elseif N == 6
        @tensor v_new[:] := OO[12, 1, 2, -1] * OO[2, 3, 4, -2] *
            OO[4, 5, 6, -3] * OO[6, 7, 8, -4] *
            OO[8, 9, 10, -5] * OO[10, 11, 12, -6] *
            v0[1, 3, 5, 7, 9, 11]
    elseif N == 8
        @tensor v_new[:] := OO[16, 1, 2, -1] * OO[2, 3, 4, -2] *
            OO[4, 5, 6, -3] * OO[6, 7, 8, -4] *
            OO[8, 9, 10, -5] * OO[10, 11, 12, -6] *
            OO[12, 13, 14, -7] * OO[14, 15, 16, -8] *
            v0[1, 3, 5, 7, 9, 11, 13, 15]
    else
        error("Dense matrix-free ES supports N=4,5,6,8.")
    end
    return kn === nothing ? v_new : dense_mf_k_projection(v_new, N, kn)
end

function _dense_mf_eigsolve(OO, N::Int, EH_n::Int;
        kn=nothing, krylovdim::Int=2 * EH_n + 5)
    v_init = dense_mf_initial_vector(OO, N)
    if kn !== nothing
        v_init = dense_mf_k_projection(v_init, N, kn)
        norm(v_init) > 1e-12 || error("Momentum projection produced a zero vector for k=$kn.")
    end

    contraction_fun(x) = CTM_T_action_dense_matrixfree(OO, x, N; kn=kn)
    eig_result = eigsolve(
        contraction_fun,
        v_init,
        EH_n,
        :LM,
        Arnoldi(krylovdim=krylovdim),
    )
    eu = ComplexF64.(eig_result[1])
    ev = eig_result[2]
    info = length(eig_result) >= 3 ? eig_result[3] : nothing
    order = sortperm(abs.(eu); rev=true)
    return eu[order], ev[order], info
end

function ES_CTMRG_ED_dense_matrixfree(
    CTM,
    U_L,
    U_R,
    D::Int,
    chi::Int,
    N::Int,
    EH_n::Int;
    save_filenm=nothing,
    T_scale=1,
    krylovdim::Int=2 * EH_n + 5,
)
    println("Dense CTM ES: matrix-free Lanczos/Arnoldi")
    println("D=$D, chi=$chi, N=$N, EH_n=$EH_n")
    OO = ES_CTMRG_prepare_dense_matrixfree(CTM, U_L, U_R; T_scale=T_scale)
    local_dim = dim(space(OO, 2))
    println("Krylov vector length = $(local_dim^N)")
    println("The $(local_dim^N) x $(local_dim^N) transfer matrix is not constructed.")
    flush(stdout)

    eu, ev, eigsolve_info = _dense_mf_eigsolve(
        OO, N, EH_n; krylovdim=krylovdim,
    )
    k_phase = dense_mf_calculate_k(ev, N)
    eu_normalized = eu / sum(eu)
    entanglement_spectrum = -log.(abs.(eu_normalized))

    if save_filenm === nothing
        save_filenm = "ES_dense_matrixfree_D$(D)_chi$(chi)_N$(N).mat"
    end
    matwrite(save_filenm, Dict(
        "eu" => eu,
        "eu_normalized" => eu_normalized,
        "entanglement_spectrum" => entanglement_spectrum,
        "k_phase" => k_phase,
        "N" => N,
        "D" => D,
        "chi" => chi,
        "T_scale" => T_scale,
        "matrix_free" => true,
    ); compress=false)
    println("Saved dense matrix-free ES to $save_filenm")
    flush(stdout)
    return (
        eu=eu,
        eu_normalized=eu_normalized,
        entanglement_spectrum=entanglement_spectrum,
        k_phase=k_phase,
        eigsolve_info=eigsolve_info,
    )
end

function ES_CTMRG_ED_Kprojector_dense_matrixfree(
    CTM,
    U_L,
    U_R,
    D::Int,
    chi::Int,
    N::Int,
    EH_n::Int;
    save_filenm=nothing,
    T_scale=1,
    krylovdim::Int=2 * EH_n + 5,
)
    println("Dense momentum-resolved CTM ES: matrix-free Lanczos/Arnoldi")
    println("D=$D, chi=$chi, N=$N, EH_n=$EH_n")
    OO = ES_CTMRG_prepare_dense_matrixfree(CTM, U_L, U_R; T_scale=T_scale)
    local_dim = dim(space(OO, 2))
    println("Krylov vector length = $(local_dim^N)")
    println("The $(local_dim^N) x $(local_dim^N) transfer matrix is not constructed.")
    flush(stdout)

    Ks = collect(0:(N - 1))
    eu_set = Matrix{Any}(undef, length(Ks), 1)
    es_set = Matrix{Any}(undef, length(Ks), 1)
    info_set = Matrix{Any}(undef, length(Ks), 1)
    for kk in eachindex(Ks)
        k = Ks[kk]
        eu, _, info = _dense_mf_eigsolve(
            OO, N, EH_n; kn=k, krylovdim=krylovdim,
        )
        eu_normalized = eu / sum(eu)
        eu_set[kk, 1] = eu
        es_set[kk, 1] = -log.(abs.(eu_normalized))
        info_set[kk, 1] = info
        println("momentum k=$k")
        println(eu)
        flush(stdout)
    end

    if save_filenm === nothing
        save_filenm = "ES_Kprojector_dense_matrixfree_D$(D)_chi$(chi)_N$(N).mat"
    end
    matwrite(save_filenm, Dict(
        "eu_set" => eu_set,
        "entanglement_spectrum_set" => es_set,
        "Ks" => Ks,
        "N" => N,
        "D" => D,
        "chi" => chi,
        "T_scale" => T_scale,
        "matrix_free" => true,
    ); compress=false)
    println("Saved dense momentum-resolved matrix-free ES to $save_filenm")
    flush(stdout)
    return (
        eu_set=eu_set,
        entanglement_spectrum_set=es_set,
        Ks=Ks,
        eigsolve_info_set=info_set,
    )
end
