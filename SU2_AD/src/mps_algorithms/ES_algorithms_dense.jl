function ES_CTMRG_prepare_dense(CTM, U_L, U_R; T_scale=1)
    Tleft = CTM.Tset.T4 / norm(CTM.Tset.T4)
    Tright = CTM.Tset.T2 / norm(CTM.Tset.T2)

    Tleft = T_scale * Tleft
    Tright = T_scale * Tright

    @tensor O1[:] := Tleft[-3, 1, -1] * U_L[1, -2, -4]
    @tensor O2[:] := Tright[-1, 1, -3] * U_R[-4, -2, 1]

    @tensor OO[:] := O1[-2, -3, -5, 1] * O2[-1, 1, -4, -6]
    U_fuse_chichi = unitary(fuse(space(OO, 1) * space(OO, 2)), space(OO, 1) * space(OO, 2))
    @tensor OO[:] := U_fuse_chichi[-1, 1, 2] * OO[1, 2, -2, 3, 4, -4] *
        U_fuse_chichi'[3, 4, -3]

    return (OO=OO,)
end

function _dense_repeated_space(V, N)
    Vout = V
    for _ in 2:N
        Vout = Vout * V
    end
    return Vout
end

function _dense_random_tensor(codomain_space)
    sz = Tuple(dim(codomain_space[n]) for n in 1:length(codomain_space))
    data = randn(Float64, sz...) + im * randn(Float64, sz...)
    return Tensor(data, codomain_space)
end

function dense_ES_initial_vector(OO, N)
    return _dense_random_tensor(_dense_repeated_space(space(OO, 2)', N))
end

function dense_translate(v, N)
    if N == 4
        return permute(v, (2, 3, 4, 1,), ())
    elseif N == 5
        return permute(v, (2, 3, 4, 5, 1,), ())
    elseif N == 6
        return permute(v, (2, 3, 4, 5, 6, 1,), ())
    elseif N == 8
        return permute(v, (2, 3, 4, 5, 6, 7, 8, 1,), ())
    else
        error("Dense ES currently supports N=4,5,6,8.")
    end
end

function dense_k_projection(v_unprojected, N, kn)
    vnorm = dot(v_unprojected, v_unprojected)
    v_work = deepcopy(v_unprojected)
    v_projected = deepcopy(v_unprojected)
    for cc in 1:(N - 1)
        v_work = dense_translate(v_work, N)
        v_projected = v_projected + exp(-im * (2 * pi * kn / N) * cc) * v_work
    end
    nrm = sqrt(dot(v_projected, v_projected))
    if abs(nrm) < 1e-12
        return v_projected
    end
    return v_projected / nrm * sqrt(vnorm)
end

function dense_calculate_k(ev, N)
    ks = Array{ComplexF64,1}(undef, length(ev))
    for cc in eachindex(ev)
        v = ev[cc]
        vp = dense_translate(v, N)
        phase = dot(vp, v) / dot(v, v)
        if N == 8
            ks[cc] = phase'
        else
            ks[cc] = phase
        end
    end
    return ks
end

function CTM_T_action_dense(OO, v0, N; kn=nothing)
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
        error("Dense ES currently supports N=4,5,6,8.")
    end
    if kn !== nothing
        v_new = dense_k_projection(v_new, N, kn)
    end
    return v_new
end

function dense_transfer_tensor(OO, N)
    if N == 4
        @tensoropt T[:] := OO[8, -5, 2, -1] * OO[2, -6, 4, -2] *
            OO[4, -7, 6, -3] * OO[6, -8, 8, -4]
    elseif N == 5
        @tensoropt T[:] := OO[10, -6, 2, -1] * OO[2, -7, 4, -2] *
            OO[4, -8, 6, -3] * OO[6, -9, 8, -4] *
            OO[8, -10, 10, -5]
    elseif N == 6
        @tensoropt T[:] := OO[12, -7, 2, -1] * OO[2, -8, 4, -2] *
            OO[4, -9, 6, -3] * OO[6, -10, 8, -4] *
            OO[8, -11, 10, -5] * OO[10, -12, 12, -6]
    elseif N == 8
        @tensoropt T[:] := OO[16, -9, 2, -1] * OO[2, -10, 4, -2] *
            OO[4, -11, 6, -3] * OO[6, -12, 8, -4] *
            OO[8, -13, 10, -5] * OO[10, -14, 12, -6] *
            OO[12, -15, 14, -7] * OO[14, -16, 16, -8]
    else
        error("Dense ES currently supports N=4,5,6,8.")
    end
    return T
end

function dense_transfer_matrix(OO, N)
    d = dim(space(OO, 2))
    T = dense_transfer_tensor(OO, N)
    return reshape(convert(Array, T), d^N, d^N)
end

function _dense_translate_vec(v::AbstractVector, d::Int, N::Int)
    data = reshape(v, ntuple(_ -> d, N))
    perm = Tuple(vcat(collect(2:N), 1))
    return vec(permutedims(data, perm))
end

function _dense_k_project_vec(v::AbstractVector, d::Int, N::Int, kn::Int)
    vnorm = dot(v, v)
    v_work = copy(v)
    v_projected = copy(v)
    for cc in 1:(N - 1)
        v_work = _dense_translate_vec(v_work, d, N)
        v_projected = v_projected + exp(-im * (2 * pi * kn / N) * cc) * v_work
    end
    nrm = sqrt(dot(v_projected, v_projected))
    if abs(nrm) < 1e-12
        return v_projected
    end
    return v_projected / nrm * sqrt(vnorm)
end

function _dense_calculate_k_vecs(evecs::AbstractMatrix, d::Int, N::Int)
    ks = Array{ComplexF64,1}(undef, size(evecs, 2))
    for cc in axes(evecs, 2)
        v = evecs[:, cc]
        vp = _dense_translate_vec(v, d, N)
        phase = dot(vp, v) / dot(v, v)
        if N == 8
            ks[cc] = phase'
        else
            ks[cc] = phase
        end
    end
    return ks
end

function _dense_top_eigensystem(M::AbstractMatrix, EH_n::Int; max_exact_dim::Int=4096)
    n = size(M, 1)
    if n <= max_exact_dim
        F = eigen(M)
        eu = ComplexF64.(F.values)
        evecs = Matrix{ComplexF64}(F.vectors)
    else
        eu_raw, ev_raw = eigsolve(M, min(EH_n, n), :LM, Arnoldi(krylovdim=EH_n * 2 + 5))
        eu = ComplexF64.(eu_raw)
        evecs = hcat([ComplexF64.(v) for v in ev_raw]...)
    end
    order = sortperm(abs.(eu); rev=true)
    keep = order[1:min(EH_n, length(order))]
    return eu[keep], evecs[:, keep]
end

function ES_CTMRG_ED_dense(CTM, U_L, U_R, D, chi, N, EH_n;
        save_filenm=nothing, T_scale=1)
    println("Dense CTM ES")
    println("D=" * string(D))
    println("chi=" * string(chi))
    println("N=" * string(N))
    flush(stdout)

    prep = ES_CTMRG_prepare_dense(CTM, U_L, U_R; T_scale=T_scale)
    local_dim = dim(space(prep.OO, 2))
    println("construct dense ES matrix, local_dim=" * string(local_dim) *
        ", target size=(" * string(local_dim^N) * ", " * string(local_dim^N) * ")")
    flush(stdout)
    M = dense_transfer_matrix(prep.OO, N)
    println("dense ES matrix size = " * string(size(M)))
    flush(stdout)
    eu, evecs = _dense_top_eigensystem(M, EH_n)
    ks = _dense_calculate_k_vecs(evecs, local_dim, N)

    ks = ComplexF64.(ks)
    eu_normalized = eu / sum(eu)
    entanglement_spectrum = -log.(abs.(eu_normalized))

    if save_filenm === nothing
        save_filenm = "ES_dense_D$(D)_chi$(chi)_N$(N).mat"
    end
    matwrite(save_filenm, Dict(
        "eu" => eu,
        "eu_normalized" => eu_normalized,
        "entanglement_spectrum" => entanglement_spectrum,
        "k_phase" => ks,
        "N" => N,
        "D" => D,
        "chi" => chi,
        "T_scale" => T_scale,
    ); compress=false)

    println("Saved dense ES to " * save_filenm)
    return (eu=eu, eu_normalized=eu_normalized, entanglement_spectrum=entanglement_spectrum, k_phase=ks)
end

function ES_CTMRG_ED_Kprojector_dense(CTM, U_L, U_R, D, chi, N, EH_n;
        save_filenm=nothing, T_scale=1)
    println("Dense CTM ES with momentum projector")
    println("D=" * string(D))
    println("chi=" * string(chi))
    println("N=" * string(N))
    flush(stdout)

    prep = ES_CTMRG_prepare_dense(CTM, U_L, U_R; T_scale=T_scale)
    local_dim = dim(space(prep.OO, 2))
    println("construct dense ES matrix, local_dim=" * string(local_dim) *
        ", target size=(" * string(local_dim^N) * ", " * string(local_dim^N) * ")")
    flush(stdout)
    M = dense_transfer_matrix(prep.OO, N)
    println("dense ES matrix size = " * string(size(M)))
    flush(stdout)

    Ks = collect(0:(N - 1))
    eu_set = Matrix{Any}(undef, length(Ks), 1)
    es_set = Matrix{Any}(undef, length(Ks), 1)

    for kk in eachindex(Ks)
        k = Ks[kk]
        v_init = randn(Float64, size(M, 1)) + im * randn(Float64, size(M, 1))
        v_init = _dense_k_project_vec(v_init, local_dim, N, k)
        if abs(norm(v_init)) < 1e-12
            eu_set[kk, 1] = ComplexF64[]
            es_set[kk, 1] = Float64[]
            continue
        end

        contraction_fun(x) = _dense_k_project_vec(M * x, local_dim, N, k)
        eu, ev = eigsolve(contraction_fun, v_init, EH_n, :LM, Arnoldi(krylovdim=EH_n * 2 + 5))
        eu = ComplexF64.(eu)
        order = sortperm(abs.(eu); rev=true)
        eu = eu[order]
        eu_norm = eu / sum(eu)
        eu_set[kk, 1] = eu
        es_set[kk, 1] = -log.(abs.(eu_norm))

        println("momentum k=" * string(k))
        println(eu)
        flush(stdout)
    end

    if save_filenm === nothing
        save_filenm = "ES_Kprojector_dense_D$(D)_chi$(chi)_N$(N).mat"
    end
    matwrite(save_filenm, Dict(
        "eu_set" => eu_set,
        "entanglement_spectrum_set" => es_set,
        "Ks" => Ks,
        "N" => N,
        "D" => D,
        "chi" => chi,
        "T_scale" => T_scale,
    ); compress=false)

    println("Saved dense momentum-resolved ES to " * save_filenm)
    return (eu_set=eu_set, entanglement_spectrum_set=es_set, Ks=Ks)
end
