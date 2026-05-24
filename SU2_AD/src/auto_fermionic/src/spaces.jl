"""
    fsector(p)

Return TensorKit's fermion-parity sector for parity `p mod 2`.
TensorKit versions in the wild accept either `FermionParity(0/1)` or
`FermionParity(false/true)`, so this helper keeps the rest of the code
version-tolerant.
"""
function fsector(p::Integer)
    q = mod(p, 2)
    try
        return FermionParity(q)
    catch
        return FermionParity(q == 1)
    end
end

function _vect_fermion_parity(args...)
    try
        return Vect[FermionParity](args...)
    catch
        return Rep[FermionParity](args...)
    end
end

"""
    fZ2space(even_dim, odd_dim; dual=false)

Fermionic graded vector space `V = V_even + V_odd`.
Unlike an ordinary bosonic `Z2Irrep`, `FermionParity` carries the nontrivial
fermionic braiding in TensorKit.
"""
function fZ2space(even_dim::Integer, odd_dim::Integer; dual::Bool=false)
    V = _vect_fermion_parity(fsector(0) => even_dim, fsector(1) => odd_dim)
    return dual ? V' : V
end

virtual_fspace(D_even::Integer, D_odd::Integer=D_even; dual::Bool=false) =
    fZ2space(D_even, D_odd; dual)

struct SpinfulBasis
    labels::NTuple{4,Tuple{Int,Int}}
end

SpinfulBasis() = SpinfulBasis(((0, 0), (1, 0), (0, 1), (1, 1)))

"""
    physical_spinful_space(; dual=false)

Four-dimensional local Fock space in the ordering
`|0>, |up>, |dn>, |up dn>`, with
`|n_up,n_dn> = (c_up')^n_up (c_dn')^n_dn |0>`.
Even states are `|0>, |up dn>` and odd states are `|up>, |dn>`.
"""
physical_spinful_space(; dual::Bool=false) = fZ2space(2, 2; dual)

