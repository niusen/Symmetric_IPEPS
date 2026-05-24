function _normalize_rdm(rho; atol=1e-14)
    traceval = real(LinearAlgebra.tr(rho))
    abs(traceval) < atol && return rho
    return rho / traceval
end

"""
    trace_spin(rho; keep=:up)

Trace one spin mode out of a one-site `4 x 4` density matrix in the basis
`|0>, |up>, |dn>, |up dn>`.
"""
function trace_spin(rho::AbstractMatrix; keep::Symbol=:up, normalize::Bool=true)
    @assert size(rho) == (4, 4)
    basis = SpinfulBasis().labels
    out = zeros(eltype(rho), 2, 2)
    keep == :up || keep == :dn || error("keep must be :up or :dn")
    keep_i = keep == :up ? 1 : 2
    trace_i = keep == :up ? 2 : 1
    for a in 1:4, b in 1:4
        basis[a][trace_i] == basis[b][trace_i] || continue
        out[basis[a][keep_i] + 1, basis[b][keep_i] + 1] += rho[a, b]
    end
    return normalize ? _normalize_rdm(out) : out
end

"""
    spin_partition_rdm(psi, nsites; keep=:up)

Reduced density matrix of all up or all down modes from a wavefunction whose
site basis is the same `4`-state basis as `SpinfulBasis`.
"""
function spin_partition_rdm(psi::AbstractVector, nsites::Integer; keep::Symbol=:up, normalize::Bool=true)
    @assert length(psi) == 4^nsites
    keep == :up || keep == :dn || error("keep must be :up or :dn")

    mode_tensor = reshape(psi, ntuple(_ -> 2, 2 * nsites))
    keep_modes = keep == :up ? collect(1:2:(2 * nsites)) : collect(2:2:(2 * nsites))
    env_modes = setdiff(collect(1:(2 * nsites)), keep_modes)
    mat = reshape(permutedims(mode_tensor, vcat(keep_modes, env_modes)), 2^nsites, 2^nsites)
    rho = mat * mat'
    return normalize ? _normalize_rdm(rho) : rho
end

function entanglement_spectrum(rho::AbstractMatrix; cutoff::Real=0)
    vals = eigvals(Hermitian((rho + rho') / 2))
    vals = sort(real(vals), rev=true)
    return vals[vals .> cutoff]
end

function entropy_vn(rho::AbstractMatrix; cutoff::Real=1e-14)
    vals = entanglement_spectrum(rho; cutoff)
    return -sum(vals .* log.(vals))
end
