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
    function truncdim(rank::Integer; kwargs...)
        return TensorKit.truncrank(rank; kwargs...)
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
