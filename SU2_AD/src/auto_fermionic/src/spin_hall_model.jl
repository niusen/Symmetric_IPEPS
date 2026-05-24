Base.@kwdef struct SpinHallParameters
    t1::ComplexF64 = 1.0 + 0.0im
    t2::ComplexF64 = 1.0 + 0.0im
    mu::ComplexF64 = 0.0 + 0.0im
    U::ComplexF64 = 0.0 + 0.0im
    mx::ComplexF64 = 0.0 + 0.0im
    B::ComplexF64 = 0.0 + 0.0im
    mx_type::String = "uniform"
end

Base.@kwdef struct SpinHallEnergySetting
    Lx::Int = 2
    Ly::Int = 2
    magnetic_cell::Int = 2
end

function _staggered_mx(mx, mx_type::String, px::Int)
    if mx_type == "uniform"
        return mx
    elseif mx_type == "x_stagger"
        return mx * (-1)^px
    else
        error("unknown mx_type: $mx_type")
    end
end

"""
    spin_hall_coefficients(setting, parameters)

Reproduce the coefficient convention used by the legacy
`Triangle_Hofstadter_Hubbard_spinHall` code.
"""
function spin_hall_coefficients(setting::SpinHallEnergySetting, p::SpinHallParameters)
    @assert mod(setting.Lx, setting.magnetic_cell) == 0

    Lx, Ly = setting.Lx, setting.Ly
    tx_up = Matrix{ComplexF64}(undef, Lx, Ly)
    ty_up = similar(tx_up)
    t2_up = similar(tx_up)
    tx_dn = similar(tx_up)
    ty_dn = similar(tx_up)
    t2_dn = similar(tx_up)
    U = similar(tx_up)
    mu = similar(tx_up)
    mx = similar(tx_up)

    for px in 1:Lx, py in 1:Ly
        tx_up[px, py] = p.t1 * im
        ty_up[px, py] = p.t1 * exp(im * (px + 1) * pi)
        t2_up[px, py] = p.t2 * exp(im * (px + 1) * pi)

        tx_dn[px, py] = conj(p.t1 * im)
        ty_dn[px, py] = -conj(p.t1 * exp(im * (px + 1) * pi))
        t2_dn[px, py] = -conj(p.t2 * exp(im * (px + 1) * pi))

        U[px, py] = p.U
        mu[px, py] = p.mu
        mx[px, py] = _staggered_mx(p.mx, p.mx_type, px)
    end

    return (
        tx_up = tx_up, ty_up = ty_up, t2_up = t2_up,
        tx_dn = tx_dn, ty_dn = ty_dn, t2_dn = t2_dn,
        U = U, mu = mu, mx = mx,
    )
end

function _mode_creation(nmodes::Int, mode::Int; T=ComplexF64)
    dim = 1 << nmodes
    cdag = zeros(T, dim, dim)
    for state in 0:(dim - 1)
        if ((state >> (mode - 1)) & 1) == 0
            parity = count_ones(state & ((1 << (mode - 1)) - 1))
            newstate = state | (1 << (mode - 1))
            cdag[newstate + 1, state + 1] = isodd(parity) ? -one(T) : one(T)
        end
    end
    return cdag
end

function _three_site_ops(; T=ComplexF64)
    nmodes = 6
    cdag = [_mode_creation(nmodes, m; T) for m in 1:nmodes]
    c = adjoint.(cdag)
    n = [cdag[m] * c[m] for m in 1:nmodes]
    return cdag, c, n
end

"""
    triangle_hamiltonian_dense(coeffs, px, py)

Dense three-site Hamiltonian for the local triangle `(LD, RD, RU)` in the
mode order `(LD up, LD dn, RD up, RD dn, RU up, RU dn)`.
"""
function triangle_hamiltonian_dense(
    coeffs,
    px::Int,
    py::Int;
    T=ComplexF64,
    hopping_weight=1,
    onsite_weight=1,
)
    cdag, c, n = _three_site_ops(; T)
    dim = size(c[1], 1)
    h = zeros(T, dim, dim)
    id = Matrix{T}(I, dim, dim)

    up(site) = 2 * site - 1
    dn(site) = 2 * site
    hop(a, b, spin, coeff) = coeff * cdag[spin(a)] * c[spin(b)] + conj(coeff) * cdag[spin(b)] * c[spin(a)]

    h .+= hop(1, 2, up, hopping_weight * coeffs.tx_up[px, py])
    h .+= hop(1, 2, dn, hopping_weight * coeffs.tx_dn[px, py])
    h .+= hop(2, 3, up, hopping_weight * coeffs.ty_up[px, py])
    h .+= hop(2, 3, dn, hopping_weight * coeffs.ty_dn[px, py])
    h .+= hop(1, 3, up, hopping_weight * coeffs.t2_up[px, py])
    h .+= hop(1, 3, dn, hopping_weight * coeffs.t2_dn[px, py])

    for site in 1:3
        nu = n[up(site)]
        nd = n[dn(site)]
        h .+= onsite_weight * (coeffs.U[px, py] / 2) .* ((nu * nd) .- 0.5 .* (nu .+ nd) .+ 0.25 .* id)
        h .-= onsite_weight * (coeffs.mu[px, py] / 2) .* (nu .+ nd)
        h .+= onsite_weight * (coeffs.mx[px, py] / 2) .* (cdag[up(site)] * c[dn(site)] + cdag[dn(site)] * c[up(site)])
    end

    return h
end

triangle_gate_dense(coeffs, px::Int, py::Int, dt::Real; kwargs...) =
    exp(-dt * triangle_hamiltonian_dense(coeffs, px, py; kwargs...))

function triangle_hamiltonian_tensormap(coeffs, px::Int, py::Int; kwargs...)
    V = physical_spinful_space()
    H = dense_multisite_to_sector_basis(triangle_hamiltonian_dense(coeffs, px, py; kwargs...), 3)
    return TensorMap(H, V * V * V, V * V * V)
end

function triangle_gate_tensormap(coeffs, px::Int, py::Int, dt::Real; kwargs...)
    V = physical_spinful_space()
    G = dense_multisite_to_sector_basis(triangle_gate_dense(coeffs, px, py, dt; kwargs...), 3)
    return TensorMap(G, V * V * V, V * V * V)
end

function spin_hall_energy_from_observables(coeffs, obs, px::Int, py::Int)
    E = coeffs.tx_up[px, py] * obs.ex_up +
        coeffs.ty_up[px, py] * obs.ey_up +
        coeffs.t2_up[px, py] * obs.e_diag_up +
        coeffs.tx_dn[px, py] * obs.ex_dn +
        coeffs.ty_dn[px, py] * obs.ey_dn +
        coeffs.t2_dn[px, py] * obs.e_diag_dn -
        coeffs.mu[px, py] * obs.e0 / 2 +
        coeffs.U[px, py] * obs.eU / 2 +
        coeffs.mx[px, py] * obs.mx / 2
    return real(E + E')
end
