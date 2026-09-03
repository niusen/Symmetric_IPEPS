"""
Observables for a bosonic SU(2)-symmetric square-lattice iPEPS cell.

This file deliberately reuses the repository CTMRG environment, the reduced
two-site density matrix used by square Full Update, and the double-layer plus
SU(2)-covariant operator-channel algorithms in `square_correl_cell.jl`.
"""

function _square_obs_cell_get(cell, cx::Int, cy::Int, Lx::Int, Ly::Int)
    x, y = mod1(cx, Lx), mod1(cy, Ly)
    return cell isa AbstractMatrix ? cell[x, y] : cell[x][y]
end

function _square_obs_scalar(value)
    value isa Number && return ComplexF64(value)
    array = convert(Array, value)
    length(array) == 1 || error("expected a scalar contraction, got size $(size(array))")
    return ComplexF64(only(array))
end

function _square_obs_normalize_boundary(boundary)
    scale = norm(boundary)
    isfinite(scale) && scale > 0 || error("zero or non-finite CTM strip boundary norm")
    return boundary / scale, log(scale)
end

function _square_obs_dense_rho2(A_set, environment, direction::Symbol, cx::Int, cy::Int)
    Lx, Ly = size(A_set)
    site1 = CartesianIndex(cx, cy)
    site2 = direction === :x ?
        CartesianIndex(mod1(cx + 1, Lx), cy) :
        CartesianIndex(cx, mod1(cy + 1, Ly))
    bond = SquareJ1CellBond(direction, site1, site2)
    A1, A2 = A_set[site1], A_set[site2]
    rho = _square_fu_two_site_density_cell(
        environment.CTM, A1, A1, A2, A2, bond, Lx, Ly,
    )
    dense = ComplexF64.(convert(Array, rho))
    ndims(dense) == 4 || error("two-site density matrix must have four indices")
    normalization = sum(
        dense[s1, s2, s1, s2]
        for s1 in axes(dense, 1), s2 in axes(dense, 2)
    )
    abs(normalization) > eps(Float64) || error("zero two-site density-matrix trace")
    return dense / normalization
end

function _square_obs_partial_trace_second(rho)
    d1, d2, d1b, d2b = size(rho)
    (d1, d2) == (d1b, d2b) || error("invalid two-site density-matrix shape")
    result = zeros(ComplexF64, d1, d1)
    for bra in 1:d1, ket in 1:d1, state2 in 1:d2
        result[bra, ket] += rho[bra, state2, ket, state2]
    end
    return result
end

function _square_obs_one_site_expectation(rho, operator)
    return sum(
        rho[bra, ket] * operator[ket, bra]
        for bra in axes(rho, 1), ket in axes(rho, 2)
    )
end

function _square_obs_two_site_expectation(rho, operator1, operator2)
    return sum(
        rho[b1, b2, k1, k2] * operator1[k1, b1] * operator2[k2, b2]
        for b1 in axes(rho, 1), b2 in axes(rho, 2),
            k1 in axes(rho, 3), k2 in axes(rho, 4)
    )
end

function square_J1_local_observables(A_set, environment)
    Lx, Ly = size(A_set)
    physical_dimension = dim(space(A_set[1, 1], 5))
    physical_dimension == 2 || error(
        "spin-1/2 observables require physical dimension 2, got $physical_dimension",
    )

    sx = ComplexF64[0 1; 1 0] / 2
    sy = ComplexF64[0 -im; im 0] / 2
    sz = ComplexF64[1 0; 0 -1] / 2
    spin_operators = (sx, sy, sz)

    rho_two_x = zeros(ComplexF64, Lx, Ly, 2, 2, 2, 2)
    rho_two_y = similar(rho_two_x)
    rho_one_from_x = zeros(ComplexF64, Lx, Ly, 2, 2)
    rho_one_from_y = similar(rho_one_from_x)
    spin_one = zeros(ComplexF64, Lx, Ly, 3)
    spin_spin_components_x = zeros(ComplexF64, Lx, Ly, 3)
    spin_spin_components_y = similar(spin_spin_components_x)

    for cx in 1:Lx, cy in 1:Ly
        rho_x = _square_obs_dense_rho2(A_set, environment, :x, cx, cy)
        rho_y = _square_obs_dense_rho2(A_set, environment, :y, cx, cy)
        rho_two_x[cx, cy, :, :, :, :] = rho_x
        rho_two_y[cx, cy, :, :, :, :] = rho_y
        rho_one_from_x[cx, cy, :, :] = _square_obs_partial_trace_second(rho_x)
        rho_one_from_y[cx, cy, :, :] = _square_obs_partial_trace_second(rho_y)
        rho_one = (
            rho_one_from_x[cx, cy, :, :] + rho_one_from_y[cx, cy, :, :]
        ) / 2
        for component in 1:3
            operator = spin_operators[component]
            spin_one[cx, cy, component] =
                _square_obs_one_site_expectation(rho_one, operator)
            spin_spin_components_x[cx, cy, component] =
                _square_obs_two_site_expectation(rho_x, operator, operator)
            spin_spin_components_y[cx, cy, component] =
                _square_obs_two_site_expectation(rho_y, operator, operator)
        end
    end

    nearest_neighbor = square_J1_energy_cell(A_set, environment; J1=1.0)
    rho_one = (rho_one_from_x + rho_one_from_y) / 2
    return (
        rho_one=rho_one,
        rho_one_from_x=rho_one_from_x,
        rho_one_from_y=rho_one_from_y,
        spin_one=spin_one,
        rho_two_x=rho_two_x,
        rho_two_y=rho_two_y,
        spin_spin_x=nearest_neighbor.Ex,
        spin_spin_y=nearest_neighbor.Ey,
        spin_spin_components_x=spin_spin_components_x,
        spin_spin_components_y=spin_spin_components_y,
        energy_per_site=nearest_neighbor.energy_per_site,
    )
end

function _square_obs_single_spin_operator(physical_space)
    # Same SU(2)-covariant Heisenberg-SVD construction as
    # `square_correl_cell.jl`, without its unused Bool-valued identity map.
    sx = ComplexF64[0 1; 1 0] / 2
    sy = ComplexF64[0 -im; im 0] / 2
    sz = ComplexF64[1 0; 0 -1] / 2
    @tensor heisenberg[:] := sx[-1, -3] * sx[-2, -4] +
        sy[-1, -3] * sy[-2, -4] + sz[-1, -3] * sz[-2, -4]
    heisenberg = TensorMap(
        heisenberg,
        physical_space ⊗ physical_space ← physical_space ⊗ physical_space,
    )
    heisenberg = permute(heisenberg, (1, 3), (2, 4))
    left, singular_values, right = tsvd(heisenberg)
    return left * singular_values, permute(right, (2, 3), (1,))
end

function _square_obs_operator_cells(A_set)
    Lx, Ly = size(A_set)
    physical_space = space(A_set[1, 1], 5)'
    spin_left, spin_right = _square_obs_single_spin_operator(physical_space)
    AA_left = Matrix{Any}(undef, Lx, Ly)
    AA_right = Matrix{Any}(undef, Lx, Ly)
    for cx in 1:Lx, cy in 1:Ly
        AA_left[cx, cy], _, _, _, _ =
            build_double_layer_extra_leg(A_set[cx, cy], spin_left)
        AA_right[cx, cy], _, _, _, _ =
            build_double_layer_extra_leg(A_set[cx, cy], spin_right)
    end
    return AA_left, AA_right
end

function _square_obs_left_boundary_x(CTM, AA, cx, cy, Lx, Ly, with_spin::Bool)
    Cset, Tset = CTM.Cset, CTM.Tset
    C1 = _square_obs_cell_get(Cset, cx - 1, cy - 1, Lx, Ly).C1
    C4 = _square_obs_cell_get(Cset, cx - 1, cy + 1, Lx, Ly).C4
    T4 = _square_obs_cell_get(Tset, cx - 1, cy, Lx, Ly).T4
    T1 = _square_obs_cell_get(Tset, cx, cy - 1, Lx, Ly).T1
    T3 = _square_obs_cell_get(Tset, cx, cy + 1, Lx, Ly).T3
    if with_spin
        @tensor boundary[:] := C1[1, 3] * T4[2, 5, 1] * C4[7, 2] *
            T1[3, 4, -1] * AA[5, 6, -2, 4, -4] * T3[-3, 6, 7]
    else
        @tensor boundary[:] := C1[1, 3] * T4[2, 5, 1] * C4[7, 2] *
            T1[3, 4, -1] * AA[5, 6, -2, 4] * T3[-3, 6, 7]
    end
    return boundary
end

function _square_obs_right_boundary_x(CTM, AA, cx, cy, Lx, Ly, with_spin::Bool)
    Cset, Tset = CTM.Cset, CTM.Tset
    T1 = _square_obs_cell_get(Tset, cx, cy - 1, Lx, Ly).T1
    T3 = _square_obs_cell_get(Tset, cx, cy + 1, Lx, Ly).T3
    C2 = _square_obs_cell_get(Cset, cx + 1, cy - 1, Lx, Ly).C2
    T2 = _square_obs_cell_get(Tset, cx + 1, cy, Lx, Ly).T2
    C3 = _square_obs_cell_get(Cset, cx + 1, cy + 1, Lx, Ly).C3
    if with_spin
        @tensor boundary[:] := T1[-1, 4, 3] * AA[-2, 6, 5, 4, -4] *
            T3[7, 6, -3] * C2[3, 1] * T2[1, 5, 2] * C3[2, 7]
    else
        @tensor boundary[:] := T1[-1, 4, 3] * AA[-2, 6, 5, 4] *
            T3[7, 6, -3] * C2[3, 1] * T2[1, 5, 2] * C3[2, 7]
    end
    return boundary
end

function _square_obs_advance_x(boundary, AA, CTM, cx, cy, Lx, Ly, mode::Symbol)
    Tset = CTM.Tset
    T1 = _square_obs_cell_get(Tset, cx, cy - 1, Lx, Ly).T1
    T3 = _square_obs_cell_get(Tset, cx, cy + 1, Lx, Ly).T3
    if mode === :identity_inactive
        @tensor next_boundary[:] := boundary[1, 3, 5] * T1[1, 2, -1] *
            AA[3, 4, -2, 2] * T3[-3, 4, 5]
    elseif mode === :identity_active
        @tensor next_boundary[:] := boundary[1, 3, 5, -4] * T1[1, 2, -1] *
            AA[3, 4, -2, 2] * T3[-3, 4, 5]
    elseif mode === :open_spin
        @tensor next_boundary[:] := boundary[1, 3, 5] * T1[1, 2, -1] *
            AA[3, 4, -2, 2, -4] * T3[-3, 4, 5]
    elseif mode === :close_spin
        @tensor next_boundary[:] := boundary[1, 3, 5, 6] * T1[1, 2, -1] *
            AA[3, 4, -2, 2, 6] * T3[-3, 4, 5]
    else
        error("unknown x-strip mode $mode")
    end
    return next_boundary
end

function _square_obs_left_boundary_y(CTM, AA, cx, cy, Lx, Ly, with_spin::Bool)
    Cset, Tset = CTM.Cset, CTM.Tset
    C2 = _square_obs_cell_get(Cset, cx + 1, cy - 1, Lx, Ly).C2
    T1 = _square_obs_cell_get(Tset, cx, cy - 1, Lx, Ly).T1
    C1 = _square_obs_cell_get(Cset, cx - 1, cy - 1, Lx, Ly).C1
    T2 = _square_obs_cell_get(Tset, cx + 1, cy, Lx, Ly).T2
    T4 = _square_obs_cell_get(Tset, cx - 1, cy, Lx, Ly).T4
    if with_spin
        @tensor boundary[:] := C2[1, 6] * T1[2, 7, 1] * C1[8, 2] *
            T2[6, 3, -1] * AA[5, -2, 3, 7, -4] * T4[-3, 5, 8]
    else
        @tensor boundary[:] := C2[1, 6] * T1[2, 7, 1] * C1[8, 2] *
            T2[6, 3, -1] * AA[5, -2, 3, 7] * T4[-3, 5, 8]
    end
    return boundary
end

function _square_obs_right_boundary_y(CTM, AA, cx, cy, Lx, Ly, with_spin::Bool)
    Cset, Tset = CTM.Cset, CTM.Tset
    C3 = _square_obs_cell_get(Cset, cx + 1, cy + 1, Lx, Ly).C3
    T3 = _square_obs_cell_get(Tset, cx, cy + 1, Lx, Ly).T3
    C4 = _square_obs_cell_get(Cset, cx - 1, cy + 1, Lx, Ly).C4
    T2 = _square_obs_cell_get(Tset, cx + 1, cy, Lx, Ly).T2
    T4 = _square_obs_cell_get(Tset, cx - 1, cy, Lx, Ly).T4
    if with_spin
        @tensor boundary[:] := C3[6, 1] * T3[1, 7, 2] * C4[2, 8] *
            T2[-1, 3, 6] * AA[5, 7, 3, -2, -4] * T4[8, 5, -3]
    else
        @tensor boundary[:] := C3[6, 1] * T3[1, 7, 2] * C4[2, 8] *
            T2[-1, 3, 6] * AA[5, 7, 3, -2] * T4[8, 5, -3]
    end
    return boundary
end

function _square_obs_advance_y(boundary, AA, CTM, cx, cy, Lx, Ly, mode::Symbol)
    Tset = CTM.Tset
    T2 = _square_obs_cell_get(Tset, cx + 1, cy, Lx, Ly).T2
    T4 = _square_obs_cell_get(Tset, cx - 1, cy, Lx, Ly).T4
    if mode === :identity_inactive
        @tensor next_boundary[:] := boundary[1, 3, 5] * T2[1, 2, -1] *
            AA[4, -2, 2, 3] * T4[-3, 4, 5]
    elseif mode === :identity_active
        @tensor next_boundary[:] := boundary[1, 3, 5, -4] * T2[1, 2, -1] *
            AA[4, -2, 2, 3] * T4[-3, 4, 5]
    elseif mode === :open_spin
        @tensor next_boundary[:] := boundary[1, 3, 5] * T2[1, 2, -1] *
            AA[4, -2, 2, 3, -4] * T4[-3, 4, 5]
    elseif mode === :close_spin
        @tensor next_boundary[:] := boundary[1, 3, 5, 6] * T2[1, 2, -1] *
            AA[4, -2, 2, 3, 6] * T4[-3, 4, 5]
    else
        error("unknown y-strip mode $mode")
    end
    return next_boundary
end

function _square_obs_left_boundary(direction, CTM, AA, cx, cy, Lx, Ly, with_spin)
    direction === :x && return _square_obs_left_boundary_x(
        CTM, AA, cx, cy, Lx, Ly, with_spin,
    )
    direction === :y && return _square_obs_left_boundary_y(
        CTM, AA, cx, cy, Lx, Ly, with_spin,
    )
    error("direction must be :x or :y")
end

function _square_obs_right_boundary(direction, CTM, AA, cx, cy, Lx, Ly, with_spin)
    direction === :x && return _square_obs_right_boundary_x(
        CTM, AA, cx, cy, Lx, Ly, with_spin,
    )
    direction === :y && return _square_obs_right_boundary_y(
        CTM, AA, cx, cy, Lx, Ly, with_spin,
    )
    error("direction must be :x or :y")
end

function _square_obs_advance(direction, boundary, AA, CTM, cx, cy, Lx, Ly, mode)
    direction === :x && return _square_obs_advance_x(
        boundary, AA, CTM, cx, cy, Lx, Ly, mode,
    )
    direction === :y && return _square_obs_advance_y(
        boundary, AA, CTM, cx, cy, Lx, Ly, mode,
    )
    error("direction must be :x or :y")
end

_square_obs_shift(cx, cy, direction::Symbol, distance::Int) =
    direction === :x ? (cx + distance, cy) : (cx, cy + distance)

function _square_obs_close_amplitude(
    direction, boundary, logscale, AA, CTM, cx, cy, Lx, Ly, with_spin,
)
    right = _square_obs_right_boundary(
        direction, CTM, AA, cx, cy, Lx, Ly, with_spin,
    )
    right, right_logscale = _square_obs_normalize_boundary(right)
    if with_spin
        @tensor overlap[:] := boundary[1, 2, 3, 4] * right[1, 2, 3, 4]
    else
        @tensor overlap[:] := boundary[1, 2, 3] * right[1, 2, 3]
    end
    return _square_obs_scalar(overlap), logscale + right_logscale
end

function _square_obs_ratio(numerator, denominator)
    numerator_scalar, numerator_logscale = numerator
    denominator_scalar, denominator_logscale = denominator
    return numerator_scalar / denominator_scalar *
        exp(numerator_logscale - denominator_logscale)
end

function _square_obs_spin_correlations(
    direction, cx, cy, distance, CTM, AA_closed, AA_left, AA_right, Lx, Ly,
)
    numerator = _square_obs_left_boundary(
        direction, CTM, AA_left[cx, cy], cx, cy, Lx, Ly, true,
    )
    numerator, numerator_logscale = _square_obs_normalize_boundary(numerator)
    denominator = _square_obs_left_boundary(
        direction, CTM, _square_obs_cell_get(AA_closed, cx, cy, Lx, Ly),
        cx, cy, Lx, Ly, false,
    )
    denominator, denominator_logscale = _square_obs_normalize_boundary(denominator)
    correlations = zeros(ComplexF64, distance)

    for separation in 1:distance
        qx, qy = _square_obs_shift(cx, cy, direction, separation)
        numerator_closed = _square_obs_close_amplitude(
            direction, numerator, numerator_logscale,
            AA_right[mod1(qx, Lx), mod1(qy, Ly)], CTM,
            qx, qy, Lx, Ly, true,
        )
        denominator_closed = _square_obs_close_amplitude(
            direction, denominator, denominator_logscale,
            _square_obs_cell_get(AA_closed, qx, qy, Lx, Ly), CTM,
            qx, qy, Lx, Ly, false,
        )
        correlations[separation] = _square_obs_ratio(
            numerator_closed, denominator_closed,
        )
        separation == distance && continue
        numerator = _square_obs_advance(
            direction, numerator,
            _square_obs_cell_get(AA_closed, qx, qy, Lx, Ly), CTM,
            qx, qy, Lx, Ly, :identity_active,
        )
        numerator, increment = _square_obs_normalize_boundary(numerator)
        numerator_logscale += increment
        denominator = _square_obs_advance(
            direction, denominator,
            _square_obs_cell_get(AA_closed, qx, qy, Lx, Ly), CTM,
            qx, qy, Lx, Ly, :identity_inactive,
        )
        denominator, increment = _square_obs_normalize_boundary(denominator)
        denominator_logscale += increment
    end
    return correlations
end

function _square_obs_dimer_correlations(
    direction, cx, cy, distance, CTM, AA_closed, AA_left, AA_right,
    local_bonds, Lx, Ly,
)
    separations = collect(2:distance)
    isempty(separations) && return (
        separations=Int[], raw=ComplexF64[], connected=ComplexF64[],
    )

    numerator = _square_obs_left_boundary(
        direction, CTM, AA_left[cx, cy], cx, cy, Lx, Ly, true,
    )
    numerator, numerator_logscale = _square_obs_normalize_boundary(numerator)
    x1, y1 = _square_obs_shift(cx, cy, direction, 1)
    numerator = _square_obs_advance(
        direction, numerator, AA_right[mod1(x1, Lx), mod1(y1, Ly)], CTM,
        x1, y1, Lx, Ly, :close_spin,
    )
    numerator, increment = _square_obs_normalize_boundary(numerator)
    numerator_logscale += increment

    denominator = _square_obs_left_boundary(
        direction, CTM, _square_obs_cell_get(AA_closed, cx, cy, Lx, Ly),
        cx, cy, Lx, Ly, false,
    )
    denominator, denominator_logscale = _square_obs_normalize_boundary(denominator)
    denominator = _square_obs_advance(
        direction, denominator,
        _square_obs_cell_get(AA_closed, x1, y1, Lx, Ly), CTM,
        x1, y1, Lx, Ly, :identity_inactive,
    )
    denominator, increment = _square_obs_normalize_boundary(denominator)
    denominator_logscale += increment

    raw = zeros(ComplexF64, length(separations))
    connected = similar(raw)
    base_bond = local_bonds[cx, cy]
    for (index, separation) in pairs(separations)
        qx, qy = _square_obs_shift(cx, cy, direction, separation)
        numerator_second = _square_obs_advance(
            direction, numerator, AA_left[mod1(qx, Lx), mod1(qy, Ly)], CTM,
            qx, qy, Lx, Ly, :open_spin,
        )
        numerator_second, numerator_increment =
            _square_obs_normalize_boundary(numerator_second)
        rx, ry = _square_obs_shift(qx, qy, direction, 1)
        numerator_closed = _square_obs_close_amplitude(
            direction, numerator_second,
            numerator_logscale + numerator_increment,
            AA_right[mod1(rx, Lx), mod1(ry, Ly)], CTM,
            rx, ry, Lx, Ly, true,
        )

        denominator_second = _square_obs_advance(
            direction, denominator,
            _square_obs_cell_get(AA_closed, qx, qy, Lx, Ly), CTM,
            qx, qy, Lx, Ly, :identity_inactive,
        )
        denominator_second, denominator_increment =
            _square_obs_normalize_boundary(denominator_second)
        denominator_closed = _square_obs_close_amplitude(
            direction, denominator_second,
            denominator_logscale + denominator_increment,
            _square_obs_cell_get(AA_closed, rx, ry, Lx, Ly), CTM,
            rx, ry, Lx, Ly, false,
        )
        raw[index] = _square_obs_ratio(numerator_closed, denominator_closed)
        remote_bond = local_bonds[mod1(qx, Lx), mod1(qy, Ly)]
        connected[index] = raw[index] - base_bond * remote_bond

        separation == last(separations) && continue
        numerator = _square_obs_advance(
            direction, numerator,
            _square_obs_cell_get(AA_closed, qx, qy, Lx, Ly), CTM,
            qx, qy, Lx, Ly, :identity_inactive,
        )
        numerator, increment = _square_obs_normalize_boundary(numerator)
        numerator_logscale += increment
        denominator = denominator_second
        denominator_logscale += denominator_increment
    end
    return (separations=separations, raw=raw, connected=connected)
end

function square_J1_correlations(
    A_set,
    environment,
    local_observables;
    distance::Int=20,
)
    distance >= 1 || error("correlation distance must be positive")
    Lx, Ly = size(A_set)
    AA_closed = environment.AA
    AA_left, AA_right = _square_obs_operator_cells(A_set)
    spin_x = zeros(ComplexF64, Lx, Ly, distance)
    spin_y = similar(spin_x)
    dimer_count = max(distance - 1, 0)
    dimer_raw_x = zeros(ComplexF64, Lx, Ly, dimer_count)
    dimer_raw_y = similar(dimer_raw_x)
    dimer_connected_x = similar(dimer_raw_x)
    dimer_connected_y = similar(dimer_raw_x)

    for cx in 1:Lx, cy in 1:Ly
        spin_x[cx, cy, :] = _square_obs_spin_correlations(
            :x, cx, cy, distance, environment.CTM,
            AA_closed, AA_left, AA_right, Lx, Ly,
        )
        spin_y[cx, cy, :] = _square_obs_spin_correlations(
            :y, cx, cy, distance, environment.CTM,
            AA_closed, AA_left, AA_right, Lx, Ly,
        )
        dimer_x = _square_obs_dimer_correlations(
            :x, cx, cy, distance, environment.CTM,
            AA_closed, AA_left, AA_right,
            local_observables.spin_spin_x, Lx, Ly,
        )
        dimer_y = _square_obs_dimer_correlations(
            :y, cx, cy, distance, environment.CTM,
            AA_closed, AA_left, AA_right,
            local_observables.spin_spin_y, Lx, Ly,
        )
        dimer_raw_x[cx, cy, :] = dimer_x.raw
        dimer_raw_y[cx, cy, :] = dimer_y.raw
        dimer_connected_x[cx, cy, :] = dimer_x.connected
        dimer_connected_y[cx, cy, :] = dimer_y.connected
    end
    return (
        spin_separations=collect(1:distance),
        spin_x=spin_x,
        spin_y=spin_y,
        dimer_separations=collect(2:distance),
        dimer_raw_x=dimer_raw_x,
        dimer_raw_y=dimer_raw_y,
        dimer_connected_x=dimer_connected_x,
        dimer_connected_y=dimer_connected_y,
    )
end

function _square_obs_transfer_action(vector, CTM, direction::Symbol, Lx, Ly, cut::Int)
    result = vector
    if direction === :x
        bottom_y = mod1(cut, Ly)
        top_y = mod1(cut + 1, Ly)
        for cx in 1:Lx
            top = _square_obs_cell_get(CTM.Tset, cx, top_y, Lx, Ly).T1
            bottom = _square_obs_cell_get(CTM.Tset, cx, bottom_y, Lx, Ly).T3
            @tensor next_result[:] := result[-1, 1, 3] *
                top[1, 2, -2] * bottom[-3, 2, 3]
            result = next_result
        end
    elseif direction === :y
        right_x = mod1(cut, Lx)
        left_x = mod1(cut + 1, Lx)
        for cy in 1:Ly
            right = _square_obs_cell_get(CTM.Tset, right_x, cy, Lx, Ly).T2
            left = _square_obs_cell_get(CTM.Tset, left_x, cy, Lx, Ly).T4
            @tensor next_result[:] := result[-1, 1, 3] *
                right[1, 2, -2] * left[-3, 2, 3]
            result = next_result
        end
    else
        error("direction must be :x or :y")
    end
    return result
end

function square_J1_transfer_spectrum(
    A_set,
    environment,
    direction::Symbol;
    n_values::Int=6,
    spins=(0, 1 / 2, 1, 3 / 2, 2),
    cut::Int=1,
)
    Lx, Ly = size(A_set)
    if direction === :x
        top = _square_obs_cell_get(environment.CTM.Tset, 1, cut + 1, Lx, Ly).T1
        bottom = _square_obs_cell_get(environment.CTM.Tset, 1, cut, Lx, Ly).T3
    elseif direction === :y
        top = _square_obs_cell_get(environment.CTM.Tset, cut, 1, Lx, Ly).T2
        bottom = _square_obs_cell_get(environment.CTM.Tset, cut + 1, 1, Lx, Ly).T4
    else
        error("direction must be :x or :y")
    end
    action(vector) = _square_obs_transfer_action(
        vector, environment.CTM, direction, Lx, Ly, cut,
    )
    eigenvalues = ComplexF64[]
    spin_labels = Float64[]
    failed_sectors = String[]
    for spin in spins
        initial = permute(
            TensorMap(
                randn,
                SU2Space(spin => 1) ⊗ space(top, 1)',
                space(bottom, 3),
            ),
            (1, 2, 3),
            (),
        )
        iszero(norm(initial)) && continue
        try
            values, _ = eigsolve(action, initial, n_values, :LM, Arnoldi())
            append!(eigenvalues, ComplexF64.(values))
            append!(spin_labels, fill(Float64(spin), length(values)))
        catch exception
            push!(failed_sectors, "spin=$spin: " * sprint(showerror, exception))
        end
    end
    isempty(eigenvalues) && error("no transfer-matrix eigenvalue was obtained")
    order = sortperm(abs.(eigenvalues); rev=true)
    eigenvalues = eigenvalues[order]
    spin_labels = spin_labels[order]
    normalized = eigenvalues / eigenvalues[1]
    period = direction === :x ? Lx : Ly
    correlation_lengths = [
        abs(abs(value) - 1) < 100eps(Float64) ? Inf :
        -period / log(abs(value))
        for value in normalized
    ]
    return (
        direction=direction,
        cell_period=period,
        cut=cut,
        eigenvalues=eigenvalues,
        normalized_eigenvalues=normalized,
        magnitudes=abs.(normalized),
        spin=spin_labels,
        correlation_lengths=correlation_lengths,
        failed_sectors=failed_sectors,
    )
end

function square_J1_analyze_observables(
    A_set,
    environment;
    distance::Int=20,
    transfer_n_values::Int=6,
    transfer_spins=(0, 1 / 2, 1, 3 / 2, 2),
)
    local_observables = square_J1_local_observables(A_set, environment)
    transfer_x = square_J1_transfer_spectrum(
        A_set, environment, :x;
        n_values=transfer_n_values, spins=transfer_spins,
    )
    transfer_y = square_J1_transfer_spectrum(
        A_set, environment, :y;
        n_values=transfer_n_values, spins=transfer_spins,
    )
    correlations = square_J1_correlations(
        A_set, environment, local_observables; distance=distance,
    )
    return (
        local_observables=local_observables,
        transfer_x=transfer_x,
        transfer_y=transfer_y,
        correlations=correlations,
    )
end
