function z2_to_fermion_space(V::GradedSpace{Z2Irrep})
    even_dim = dim(V, Z2Irrep(0))
    odd_dim = dim(V, Z2Irrep(1))
    return fZ2space(even_dim, odd_dim; dual=isdual(V))
end

function z2_to_fermion_product_space(P::ProductSpace)
    spaces = map(z2_to_fermion_space, P.spaces)
    isempty(spaces) && return ProductSpace{typeof(fZ2space(0, 0))}(())
    out = first(spaces)
    for V in Iterators.drop(spaces, 1)
        out = out * V
    end
    return out
end

"""
    z2_to_fermion_tensor(T)

Convert a legacy ordinary-Z2 TensorKit tensor into the corresponding
`FermionParity` tensor by preserving dense sector-ordered data and replacing
each leg space. This is a representation conversion only; it does not insert
or remove any explicit legacy swap gates.
"""
function z2_to_fermion_tensor(T::TensorKit.AbstractTensorMap)
    cod = z2_to_fermion_product_space(codomain(T))
    dom = z2_to_fermion_product_space(domain(T))
    return TensorMap(convert(Array, T), cod, dom)
end

function z2_to_fermion_pess(B, T)
    return FermionicTrianglePESS(z2_to_fermion_tensor(B), z2_to_fermion_tensor(T))
end

function z2_to_fermion_pess(x)
    hasproperty(x, :Bm) && hasproperty(x, :Tm) || error("object does not look like a Triangle_iPESS")
    return z2_to_fermion_pess(getproperty(x, :Bm), getproperty(x, :Tm))
end

function z2_to_fermion_state(x::AbstractMatrix)
    out = Matrix{FermionicTrianglePESS}(undef, size(x)...)
    for i in eachindex(x)
        out[i] = z2_to_fermion_pess(x[i])
    end
    return out
end

"""
    legacy_z2_pess_to_fermion_ipeps_tensor(x)

Convert a legacy saved `Triangle_iPESS`-like object to a fermionic iPEPS tensor
in the legacy square-tensor convention.  This intentionally performs the old
bosonic/Z2 PESS-to-iPEPS contraction first and only then replaces `Z2Irrep` by
`FermionParity`.

Use this for old saved states.  It preserves the old swap-gate convention better
than converting `B` and `T` separately and then contracting them with graded
permutation rules.
"""
function legacy_z2_pess_to_fermion_ipeps_tensor(x)
    hasproperty(x, :Bm) && hasproperty(x, :Tm) || error("object does not look like a Triangle_iPESS")
    A = permute(getproperty(x, :Tm) * getproperty(x, :Bm), (1, 5, 4, 2, 3))
    return z2_to_fermion_tensor(A)
end

function legacy_z2_state_to_fermion_ipeps_cell(x::AbstractMatrix)
    out = Matrix{Any}(undef, size(x)...)
    for i in eachindex(x)
        out[i] = legacy_z2_pess_to_fermion_ipeps_tensor(x[i])
    end
    return out
end

conversion_error(T::TensorMap, F::TensorMap) =
    norm(convert(Array, T) - convert(Array, F)) / max(norm(convert(Array, T)), eps())
