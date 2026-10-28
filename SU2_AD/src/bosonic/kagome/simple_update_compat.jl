# Included ONLY inside KagomeSU2Cells.Legacy. These local wrappers follow the
# old/new TensorKit dispatch used by dmrg/src/settings.jl, without adding
# methods to TensorKit or changing any caller's global method table.
# The two SVD truncations still count total states (2j+1 per multiplet), and
# the original Truncations helper still removes degeneracies at the boundary.

# Declare local bindings before `using TensorKit`, avoiding method piracy.
function permute end
function tsvd end
function truncdim end
function truncerr end
function eigh end

import TensorKit
using TensorKit: @tensor, TensorMap, SU2Space, SU2Irrep, GradedSpace, ComplexSpace,
                 CartesianSpace, space, sectors, blocksectors, dim, unitary, fuse,
                 domain, codomain, ⊗, ←
using LinearAlgebra, ChainRulesCore

permute(t, out::Tuple, incoming::Tuple=();kwargs...) =
    TensorKit.permute(t,(out,incoming);kwargs...)
permute(t, p::Tuple{<:Tuple,<:Tuple};kwargs...) = TensorKit.permute(t,p;kwargs...)

function tsvd(t;kwargs...)
    if isdefined(TensorKit,:tsvd)
        return TensorKit.tsvd(t;kwargs...)
    end
    return TensorKit.svd_trunc(t;kwargs...)
end
tsvd(t,out::Tuple,incoming::Tuple;kwargs...) = tsvd(permute(t,out,incoming);kwargs...)
tsvd(t,p::Tuple{<:Tuple,<:Tuple};kwargs...) = tsvd(permute(t,p);kwargs...)

truncdim(rank::Integer) = isdefined(TensorKit,:truncdim) ?
    TensorKit.truncdim(rank) : TensorKit.truncrank(rank)
truncerr(tolerance::Real) = isdefined(TensorKit,:truncerr) ?
    TensorKit.truncerr(tolerance) : TensorKit.truncerror(;atol=tolerance)
eigh(t) = isdefined(TensorKit,:eigh) ? TensorKit.eigh(t) : TensorKit.eigh_full(t)
