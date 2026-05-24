struct DenseSpinfulOperators{T}
    id::Matrix{T}
    n_up::Matrix{T}
    n_dn::Matrix{T}
    n_total::Matrix{T}
    n_double::Matrix{T}
    cdag_up::Matrix{T}
    c_up::Matrix{T}
    cdag_dn::Matrix{T}
    c_dn::Matrix{T}
    sx::Matrix{T}
    sy::Matrix{T}
    sz::Matrix{T}
end

const SPINFUL_SECTOR_ORDER = (1, 4, 2, 3)

dense_to_sector_basis(M::AbstractMatrix) =
    M[collect(SPINFUL_SECTOR_ORDER), collect(SPINFUL_SECTOR_ORDER)]

function sector_to_dense_basis(M::AbstractMatrix)
    inv_order = invperm(collect(SPINFUL_SECTOR_ORDER))
    return M[inv_order, inv_order]
end

function multisite_sector_order(nsites::Integer)
    nsites > 0 || throw(ArgumentError("nsites must be positive"))
    single = collect(SPINFUL_SECTOR_ORDER)
    order = Vector{Int}(undef, 4^nsites)
    for new_state in 0:(4^nsites - 1)
        old_state = 0
        rest = new_state
        for site in 1:nsites
            sector_local = (rest % 4) + 1
            old_local = single[sector_local] - 1
            old_state += old_local * 4^(site - 1)
            rest ÷= 4
        end
        order[new_state + 1] = old_state + 1
    end
    return order
end

dense_multisite_to_sector_basis(M::AbstractMatrix, nsites::Integer) =
    M[multisite_sector_order(nsites), multisite_sector_order(nsites)]

function sector_multisite_to_dense_basis(M::AbstractMatrix, nsites::Integer)
    inv_order = invperm(multisite_sector_order(nsites))
    return M[inv_order, inv_order]
end

"""
    dense_spinful_operators([T=ComplexF64])

Dense local fermion operators for the fixed physical ordering documented in
`physical_spinful_space`. The down creation operator contains the Jordan-Wigner
sign from crossing the up mode on the same site.
"""
function dense_spinful_operators(::Type{T}=ComplexF64) where {T}
    id = Matrix{T}(I, 4, 4)
    z = zeros(T, 4, 4)

    cdag_up = copy(z)
    cdag_up[2, 1] = one(T)
    cdag_up[4, 3] = one(T)

    cdag_dn = copy(z)
    cdag_dn[3, 1] = one(T)
    cdag_dn[4, 2] = -one(T)

    c_up = Matrix(cdag_up')
    c_dn = Matrix(cdag_dn')
    n_up = cdag_up * c_up
    n_dn = cdag_dn * c_dn
    n_total = n_up + n_dn
    n_double = n_up * n_dn

    sx = cdag_up * c_dn + cdag_dn * c_up
    sy = -im * cdag_up * c_dn + im * cdag_dn * c_up
    sz = n_up - n_dn

    return DenseSpinfulOperators(
        id, n_up, n_dn, n_total, n_double,
        cdag_up, c_up, cdag_dn, c_dn, sx, sy, sz,
    )
end

"""
    tensorkit_spinful_operators()

Build TensorKit maps on the fermionic physical space. Even local observables are
ordinary maps `V -> V`. Odd `c` and `cdag` operators are returned with an
auxiliary odd parity leg, which is the standard graded way to represent parity
changing operators as even tensor maps.
"""
function tensorkit_spinful_operators(; T=ComplexF64)
    V = physical_spinful_space()
    Fodd = fZ2space(0, 1)
    ops = dense_spinful_operators(T)

    # TensorKit stores the graded physical basis by sector:
    # even states first, then odd states. Our dense physics basis is
    # |0>, |up>, |dn>, |up dn>, so we reorder to |0>, |up dn>, |up>, |dn>
    # before constructing TensorMaps.
    even_map(M) = TensorMap(dense_to_sector_basis(M), V, V)

    function odd_left_map(M)
        data = zeros(T, 1, 4, 4)
        data[1, :, :] .= dense_to_sector_basis(M)
        return TensorMap(data, Fodd * V, V)
    end

    return (
        V = V,
        Fodd = Fodd,
        id = even_map(ops.id),
        n_up = even_map(ops.n_up),
        n_dn = even_map(ops.n_dn),
        n_total = even_map(ops.n_total),
        n_double = even_map(ops.n_double),
        sx = even_map(ops.sx),
        sy = even_map(ops.sy),
        sz = even_map(ops.sz),
        cdag_up = odd_left_map(ops.cdag_up),
        c_up = odd_left_map(ops.c_up),
        cdag_dn = odd_left_map(ops.cdag_dn),
        c_dn = odd_left_map(ops.c_dn),
    )
end
