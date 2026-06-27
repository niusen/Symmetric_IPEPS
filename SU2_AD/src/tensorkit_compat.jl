function TensorKit.TensorMap(::typeof(randn), codomain_space, domain_space)
    return randn(Float64, codomain_space, domain_space)
end

function TensorKit.TensorMap(::typeof(randn), tensor_map_space)
    return randn(Float64, tensor_map_space)
end

function TensorKit.permute(
    t::TensorKit.AbstractTensorMap,
    codomain_inds::Tuple{Vararg{Int}},
    domain_inds::Tuple{Vararg{Int}};
    kwargs...,
)
    return TensorKit.permute(t, (codomain_inds, domain_inds); kwargs...)
end

if !isdefined(@__MODULE__, :truncdim)
    function truncdim(rank::Integer; multiplet_tol=nothing, kwargs...)
        if isnothing(multiplet_tol)
            return TensorKit.truncrank(rank; kwargs...)
        else
            return TensorKit.truncmultiplet(rank; multiplet_tol=multiplet_tol, kwargs...)
        end
    end
end

if !isdefined(@__MODULE__, :truncerr)
    function truncerr(epsilon::Real; kwargs...)
        return TensorKit.truncerror(; atol=epsilon, kwargs...)
    end
end

if !isdefined(@__MODULE__, :truncbelow)
    function truncbelow(epsilon::Real; kwargs...)
        return TensorKit.trunctol(; atol=epsilon, kwargs...)
    end
end

_tensorkit_svd3(result) = length(result) == 4 ? (result[1], result[2], result[3]) : result
_tensorkit_mak() = getfield(TensorKit, :MatrixAlgebraKit)

function _copy_svd_block_to_output!(Udst, Sdst, Vdst, Usrc, Ssrc, Vsrc)
    copyto!(Udst, Usrc)
    MAK = _tensorkit_mak()
    copyto!(MAK.diagview(Sdst), MAK.diagview(Ssrc))
    copyto!(Vdst, Vsrc)
    return nothing
end

function _safe_svd_compact_blocks!(
    A::TensorKit.AbstractTensorMap,
    source::TensorKit.AbstractTensorMap,
    USV,
    alg,
)
    MAK = _tensorkit_mak()
    U, S, V = USV
    for (sector, Ablock) in TensorKit.blocks(A)
        Ublock = TensorKit.block(U, sector)
        Sblock = TensorKit.block(S, sector)
        Vblock = TensorKit.block(V, sector)
        try
            MAK.svd_compact!(Ablock, (Ublock, Sblock, Vblock), alg)
        catch err
            println(stderr, "tsvd_safe_gpu: GPU block SVD failed at sector ", sector,
                ", matrix size = ", size(Ablock),
                ", storage = ", typeof(Ablock),
                ". Falling back to CPU for this block.")
            println(stderr, "tsvd_safe_gpu: error type = ", typeof(err))
            println(stderr, "tsvd_safe_gpu: error = ", err)
            flush(stderr)
            Ablock_cpu = Array(TensorKit.block(source, sector))
            Ucpu, Scpu, Vcpu = MAK.svd_compact(Ablock_cpu)
            _copy_svd_block_to_output!(Ublock, Sblock, Vblock, Ucpu, Scpu, Vcpu)
            Ablock_cpu = nothing
            Ucpu = nothing
            Scpu = nothing
            Vcpu = nothing
            if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
                ipeps_reclaim_device_memory!()
            end
        end
    end
    return USV
end

function _tensorkit_svd_trunc_safe_gpu(t::TensorKit.AbstractTensorMap; kwargs...)
    MAK = _tensorkit_mak()
    A = MAK.copy_input(TensorKit.svd_trunc, t)
    alg = MAK.select_algorithm(TensorKit.svd_trunc!, A, nothing; kwargs...)
    USV = MAK.initialize_output(TensorKit.svd_trunc!, A, alg)
    _safe_svd_compact_blocks!(A, t, USV, alg.alg)
    USV_trunc, ind = MAK.truncate(TensorKit.svd_trunc!, USV, alg.trunc)
    err = MAK.truncation_error!(MAK.diagview(USV[2]), ind)
    A = nothing
    USV = nothing
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!()
    end
    return (USV_trunc..., err)
end

function tsvd_safe_gpu(t::TensorKit.AbstractTensorMap; kwargs...)
    return _tensorkit_svd3(_tensorkit_svd_trunc_safe_gpu(t; kwargs...))
end

function tsvd_safe_gpu(t::TensorKit.AbstractTensorMap, p::Tuple{<:Tuple, <:Tuple}; kwargs...)
    return tsvd_safe_gpu(TensorKit.permute(t, p); kwargs...)
end

function tsvd_safe_gpu(
    t::TensorKit.AbstractTensorMap,
    codomain_inds::Tuple{Vararg{Int}},
    domain_inds::Tuple{Vararg{Int}};
    kwargs...,
)
    return tsvd_safe_gpu(TensorKit.permute(t, (codomain_inds, domain_inds)); kwargs...)
end

if !isdefined(@__MODULE__, :tsvd)
    function tsvd(t::TensorKit.AbstractTensorMap; kwargs...)
        return _tensorkit_svd3(TensorKit.svd_trunc(t; kwargs...))
    end

    function tsvd(t::TensorKit.AbstractTensorMap, p::Tuple{<:Tuple, <:Tuple}; kwargs...)
        return tsvd(TensorKit.permute(t, p); kwargs...)
    end

    function tsvd(
        t::TensorKit.AbstractTensorMap,
        codomain_inds::Tuple{Vararg{Int}},
        domain_inds::Tuple{Vararg{Int}};
        kwargs...,
    )
        return tsvd(TensorKit.permute(t, (codomain_inds, domain_inds)); kwargs...)
    end
end

if !isdefined(@__MODULE__, :eigh)
    function eigh(t::TensorKit.AbstractTensorMap; kwargs...)
        return TensorKit.eigh_full(t; kwargs...)
    end
end

if !isdefined(@__MODULE__, :eig)
    function eig(t::TensorKit.AbstractTensorMap; kwargs...)
        return TensorKit.eig_full(t; kwargs...)
    end

    function eig(t::TensorKit.AbstractTensorMap, p::Tuple{<:Tuple, <:Tuple}; kwargs...)
        return eig(TensorKit.permute(t, p); kwargs...)
    end

    function eig(
        t::TensorKit.AbstractTensorMap,
        codomain_inds::Tuple{Vararg{Int}},
        domain_inds::Tuple{Vararg{Int}};
        kwargs...,
    )
        return eig(TensorKit.permute(t, (codomain_inds, domain_inds)); kwargs...)
    end
end

if !isdefined(@__MODULE__, :leftorth)
    leftorth(args...; kwargs...) = TensorKit.left_orth(args...; kwargs...)
end

if !isdefined(@__MODULE__, :rightorth)
    rightorth(args...; kwargs...) = TensorKit.right_orth(args...; kwargs...)
end

if !isdefined(@__MODULE__, :leftnull)
    leftnull(args...; kwargs...) = TensorKit.left_null(args...; kwargs...)
end

if !isdefined(@__MODULE__, :rightnull)
    rightnull(args...; kwargs...) = TensorKit.right_null(args...; kwargs...)
end
