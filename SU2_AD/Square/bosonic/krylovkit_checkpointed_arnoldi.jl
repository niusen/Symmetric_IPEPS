using JLD2
using LinearAlgebra: I, lmul!, rmul!

_krylov_warn_level() =
    isdefined(KrylovKit, :WARN_LEVEL) ? getfield(KrylovKit, :WARN_LEVEL) : 1

_krylov_eachiteration_level() =
    isdefined(KrylovKit, :EACHITERATION_LEVEL) ? getfield(KrylovKit, :EACHITERATION_LEVEL) : 3

function _checkpoint_restorearnoldiform!(U, H, f, keep)
    if isdefined(KrylovKit, :_restorearnoldiform!)
        return getfield(KrylovKit, :_restorearnoldiform!)(U, H, f, keep)
    end
    isdefined(KrylovKit, :householder) ||
        error("This KrylovKit version does not expose householder; cannot restore Arnoldi form for checkpointed restart.")
    householder_fun = getfield(KrylovKit, :householder)
    @inbounds for j in 1:keep
        H[keep + 1, j] = conj(f[j])
    end
    @inbounds for j in keep:-1:1
        h, nu = householder_fun(H, j + 1, 1:j, j)
        H[j + 1, j] = nu
        H[j + 1, 1:(j - 1)] .= 0
        lmul!(h, H)
        rmul!(view(H, 1:j, :), h')
        rmul!(U, h')
    end
    return nothing
end

function _jldsave_atomic(filenm::AbstractString; kwargs...)
    tmp_filenm = filenm * ".tmp.jld2"
    jldsave(tmp_filenm; kwargs...)
    mv(tmp_filenm, filenm; force=true)
    return nothing
end

_krylov_basis_filenm(dir::AbstractString, i::Int) =
    joinpath(dir, "basis_" * lpad(string(i), 6, "0") * ".jld2")

function _krylov_cleanup_stale_basis!(dir::AbstractString, k::Int)
    isdir(dir) || return nothing
    for filenm in readdir(dir)
        m = match(r"^basis_(\d+)\.jld2$", filenm)
        if m !== nothing && parse(Int, m.captures[1]) > k
            rm(joinpath(dir, filenm); force=true)
        end
    end
    return nothing
end

function _krylov_checkpoint_save(dir::AbstractString, fact, numiter::Int, numops::Int;
                                 stage::AbstractString, howmany::Int, which, alg,
                                 basis_mode::Symbol=:none,
                                 verbose::Bool=true)
    mkpath(dir)
    k = length(fact)
    B = KrylovKit.basis(fact)

    if basis_mode === :all
        for i in 1:k
            verbose && println("  checkpoint save basis vector ", i, "/", k,
                " -> ", _krylov_basis_filenm(dir, i))
            _jldsave_atomic(_krylov_basis_filenm(dir, i); v=B[i])
        end
        _krylov_cleanup_stale_basis!(dir, k)
    elseif basis_mode === :latest
        verbose && println("  checkpoint save latest basis vector ", k,
            " -> ", _krylov_basis_filenm(dir, k))
        _jldsave_atomic(_krylov_basis_filenm(dir, k); v=B[k])
        _krylov_cleanup_stale_basis!(dir, k)
    elseif basis_mode === :none
        nothing
    else
        error("unknown basis_mode=$basis_mode")
    end

    if basis_mode !== :none
        verbose && println("  checkpoint save residual -> ", joinpath(dir, "residual.jld2"))
        _jldsave_atomic(joinpath(dir, "residual.jld2"); r=KrylovKit.residual(fact))
    end

    # Write metadata last: this is the commit marker for a complete checkpoint.
    _jldsave_atomic(joinpath(dir, "meta.jld2");
        k=k,
        H=copy(fact.H),
        numiter=numiter,
        numops=numops,
        stage=stage,
        howmany=howmany,
        which=which,
        krylovdim=alg.krylovdim,
        maxiter=alg.maxiter,
        tol=alg.tol,
        eager=alg.eager,
    )
    verbose && println("Arnoldi checkpoint saved: dir=", dir,
        ", stage=", stage,
        ", krylov_vectors=", k,
        ", numops=", numops,
        ", numiter=", numiter,
        ", basis_mode=", basis_mode)
    return nothing
end

function _krylov_checkpoint_load(dir::AbstractString; verbose::Bool=true)
    data = load(joinpath(dir, "meta.jld2"))
    k = data["k"]
    verbose && println("Loading Arnoldi checkpoint from ", dir,
        ": stage=", data["stage"],
        ", krylov_vectors=", k,
        ", numops=", data["numops"],
        ", numiter=", data["numiter"])
    first_vector = load(_krylov_basis_filenm(dir, 1))["v"]
    vectors = Vector{typeof(first_vector)}(undef, k)
    vectors[1] = first_vector
    verbose && println("  loaded basis vector 1/", k,
        " <- ", _krylov_basis_filenm(dir, 1))
    for i in 2:k
        vectors[i] = load(_krylov_basis_filenm(dir, i))["v"]
        verbose && println("  loaded basis vector ", i, "/", k,
            " <- ", _krylov_basis_filenm(dir, i))
    end
    r = load(joinpath(dir, "residual.jld2"))["r"]
    verbose && println("  loaded residual <- ", joinpath(dir, "residual.jld2"))
    V = KrylovKit.OrthonormalBasis(vectors)
    fact = KrylovKit.ArnoldiFactorization(k, V, data["H"], r)
    return fact, data["numiter"], data["numops"]
end

"""
    checkpointed_arnoldi_eigsolve(A, x0, howmany, which; kwargs...)

KrylovKit Arnoldi eigsolve with restart checkpoints.  The checkpoint stores the
full Arnoldi factorization (`basis`, compact Hessenberg data and residual), not
only the latest vector, because Krylov-Schur restarts rotate and shrink the
basis.  Resuming from the checkpoint avoids recomputing all previous completed
matrix-vector applications.

This is intended for very expensive transfer-matrix actions.  If the job is
interrupted during one call to `A(v)`, that in-flight application still has to be
redone; after each completed Arnoldi expansion or restart the state is saved.
"""
function checkpointed_arnoldi_eigsolve(A, x0, howmany::Int=1, which=:LM;
        krylovdim::Int=30,
        maxiter::Int=100,
        tol::Real=1e-12,
        orth=KrylovKit.KrylovDefaults.orth,
        eager::Bool=false,
        verbosity::Int=_krylov_warn_level(),
        checkpoint_file::AbstractString="krylov_arnoldi_checkpoint",
        checkpoint_dir=nothing,
        checkpoint_every::Int=1,
        resume::Bool=true,
        checkpoint_verbose::Bool=true,
        relative_tol::Bool=true)

    alg = KrylovKit.Arnoldi(;
        krylovdim=krylovdim,
        maxiter=maxiter,
        tol=tol,
        orth=orth,
        eager=eager,
        verbosity=verbosity,
    )
    return checkpointed_arnoldi_eigsolve(A, x0, howmany, which, alg;
        checkpoint_file=checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        resume=resume,
        checkpoint_verbose=checkpoint_verbose,
        relative_tol=relative_tol,
    )
end

function checkpointed_arnoldi_eigsolve(A, x0, howmany::Int, which, alg::KrylovKit.Arnoldi;
        checkpoint_file::AbstractString="krylov_arnoldi_checkpoint",
        checkpoint_dir=nothing,
        checkpoint_every::Int=1,
        resume::Bool=true,
        checkpoint_verbose::Bool=true,
        relative_tol::Bool=true)

    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    checkpoint_every >= 1 || error("checkpoint_every must be positive")
    KrylovKit.checkwhich(which) || error("Unknown eigenvalue selector: which = $which")

    iter = KrylovKit.ArnoldiIterator(A, x0, alg.orth)
    ckpt_dir = checkpoint_dir === nothing ? replace(checkpoint_file, r"\.jld2$" => "") : checkpoint_dir
    meta_file = joinpath(ckpt_dir, "meta.jld2")

    if resume && isfile(meta_file)
        fact, numiter, numops = _krylov_checkpoint_load(ckpt_dir; verbose=checkpoint_verbose)
        sizehint!(fact, krylovdim)
        println("Resuming Arnoldi checkpoint from ", ckpt_dir,
            ": numiter=", numiter, ", numops=", numops,
            ", krylov length=", length(fact))
    else
        numiter = 1
        fact = KrylovKit.initialize(iter; verbosity=alg.verbosity)
        numops = 1
        sizehint!(fact, krylovdim)
        _krylov_checkpoint_save(ckpt_dir, fact, numiter, numops;
            stage="initialize", howmany=howmany, which=which, alg=alg,
            basis_mode=:all, verbose=checkpoint_verbose)
    end

    beta = KrylovKit.normres(fact)
    tol_value::eltype(beta) = alg.tol

    HH = fill(zero(eltype(fact)), krylovdim, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)
    ff = fill(zero(eltype(fact)), krylovdim)

    converged = 0
    local T, U, f

    while true
        beta = KrylovKit.normres(fact)
        K = length(fact)

        if beta <= tol_value && K < howmany
            if alg.verbosity >= _krylov_warn_level()
                @warn "Invariant subspace of dimension $K is smaller than howmany=$howmany."
            end
        end

        if K == krylovdim || beta <= tol_value || (alg.eager && K >= howmany)
            H = view(HH, 1:K, 1:K)
            U = view(UU, 1:K, 1:K)
            f = view(ff, 1:K)
            copyto!(U, I)
            copyto!(H, KrylovKit.rayleighquotient(fact))

            T, U, values = KrylovKit.hschur!(H, U)
            by, rev = KrylovKit.eigsort(which)
            p = sortperm(values; by=by, rev=rev)
            T, U = KrylovKit.permuteschur!(T, U, p)
            f .= conj.(view(U, K, :)) .* beta

            converged = 0
            while converged < length(fact)
                residual_i = abs(f[converged + 1])
                threshold_i = relative_tol ?
                    tol_value * max(one(residual_i), abs(values[converged + 1])) :
                    tol_value
                residual_i <= threshold_i || break
                converged += 1
            end
            if checkpoint_verbose
                nshow = min(howmany, length(fact))
                for ii in 1:nshow
                    residual_i = abs(f[ii])
                    threshold_i = relative_tol ?
                        tol_value * max(one(residual_i), abs(values[ii])) :
                        tol_value
                    println("  Arnoldi convergence check ", ii,
                        ": residual=", residual_i,
                        ", threshold=", threshold_i,
                        ", relative_tol=", relative_tol)
                end
            end
            if 0 < converged < length(fact) && !iszero(T[converged + 1, converged])
                converged -= 1
            end

            _krylov_checkpoint_save(ckpt_dir, fact, numiter, numops;
                stage="processed", howmany=howmany, which=which, alg=alg,
                basis_mode=:none, verbose=checkpoint_verbose)

            if converged >= howmany || beta <= tol_value
                break
            elseif alg.verbosity >= _krylov_eachiteration_level()
                @info "Checkpointed Arnoldi iteration $numiter, step=$K: $converged values converged"
            end
        end

        if K < krylovdim
            fact = KrylovKit.expand!(iter, fact; verbosity=alg.verbosity)
            numops += 1
            if numops % checkpoint_every == 0
                _krylov_checkpoint_save(ckpt_dir, fact, numiter, numops;
                    stage="expand", howmany=howmany, which=which, alg=alg,
                    basis_mode=:latest, verbose=checkpoint_verbose)
            end
        else
            numiter == maxiter && break

            keep = div(3 * krylovdim + 2 * converged, 5)
            if checkpoint_verbose
                nshow = min(max(howmany, 1), length(fact))
                println("Arnoldi restart required: numiter=", numiter,
                    ", numops=", numops,
                    ", krylov_vectors=", K,
                    ", converged=", converged,
                    "/", howmany,
                    ", keep=", keep)
                for ii in 1:nshow
                    residual_i = abs(f[ii])
                    threshold_i = relative_tol ?
                        tol_value * max(one(residual_i), abs(values[ii])) :
                        tol_value
                    println("  before restart eig ", ii,
                        ": value=", values[ii],
                        ", residual=", residual_i,
                        ", threshold=", threshold_i,
                        ", ratio=", residual_i / threshold_i)
                end
            end
            if !iszero(T[keep + 1, keep])
                if keep > 1
                    keep -= 1
                else
                    keep += 1
                    if krylovdim == 2
                        alg.verbosity >= _krylov_warn_level() &&
                            @warn "Arnoldi iteration got stuck in a 2x2 block; increase krylovdim"
                        break
                    end
                end
            end

            _checkpoint_restorearnoldiform!(U, H, f, keep)
            copy!(KrylovKit.rayleighquotient(fact), H)
            B = KrylovKit.basis(fact)
            KrylovKit.basistransform!(B, view(U, :, 1:keep))
            B[keep + 1] = KrylovKit.scale!!(KrylovKit.residual(fact), 1 / beta)
            fact = KrylovKit.shrink!(fact, keep)
            numiter += 1

            _krylov_checkpoint_save(ckpt_dir, fact, numiter, numops;
                stage="restart", howmany=howmany, which=which, alg=alg,
                basis_mode=:all, verbose=checkpoint_verbose)
        end
    end

    howmany_out = howmany
    if eltype(T) <: Real && howmany < length(fact) && T[howmany + 1, howmany] != 0
        howmany_out += 1
    elseif size(T, 1) < howmany
        howmany_out = size(T, 1)
    end
    if converged > howmany_out
        howmany_out = converged
    end

    TT = view(T, 1:howmany_out, 1:howmany_out)
    values = KrylovKit.schur2eigvals(TT)
    eigvec_coeffs = view(U, :, 1:howmany_out) * KrylovKit.schur2eigvecs(TT)

    vectors = let B = KrylovKit.basis(fact)
        [B * v for v in eachcol(eigvec_coeffs)]
    end
    residuals = let r = KrylovKit.residual(fact)
        [KrylovKit.scale(r, last(v)) for v in eachcol(eigvec_coeffs)]
    end
    normresiduals = [KrylovKit.normres(fact) * abs(last(v)) for v in eachcol(eigvec_coeffs)]

    info = KrylovKit.ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
    _krylov_checkpoint_save(ckpt_dir, fact, numiter, numops;
        stage="finished", howmany=howmany, which=which, alg=alg,
        basis_mode=:none, verbose=checkpoint_verbose)

    return values, vectors, info
end
