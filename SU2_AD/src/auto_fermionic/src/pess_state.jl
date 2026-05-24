struct FermionicTrianglePESS{TB,TT}
    B::TB
    T::TT
end

function _rand_tensormap(cod, dom; complex_entries::Bool=true)
    if complex_entries
        return randn(ComplexF64, cod, dom)
    else
        return randn(Float64, cod, dom)
    end
end

"""
    random_triangle_pess(D_even, D_odd=D_even; rng=Random.default_rng())

Create one graded spinful triangular PESS tensor pair. The leg convention is
kept close to the legacy code:

- `B`: map from two virtual legs to `(virtual dual, physical)`.
- `T`: map from one virtual dual leg to two virtual legs.

The physical leg is `physical_spinful_space()`, i.e. a genuine
`FermionParity`-graded four-state spinful Fock space.
"""
function random_triangle_pess(
    D_even::Integer,
    D_odd::Integer=D_even;
    complex_entries::Bool=true,
)
    V = virtual_fspace(D_even, D_odd)
    Vp = physical_spinful_space()
    B = _rand_tensormap(V' * Vp, V * V; complex_entries)
    T = _rand_tensormap(V * V, V'; complex_entries)
    return FermionicTrianglePESS(B, T)
end

"""
    pess_to_ipeps_tensor(pess)

Contract the triangular PESS pair into a square-lattice iPEPS tensor using
TensorKit's graded `permute` and multiplication. The returned tensor has the
legacy order `(L, D, R, U, physical)`.
"""
function pess_to_ipeps_tensor(pess::FermionicTrianglePESS)
    Tm = permute(pess.T, ((1, 2), (3,)))
    Bm = permute(pess.B, ((1,), (2, 3, 4)))
    return permute(Tm * Bm, (1, 5, 4, 2, 3))
end
