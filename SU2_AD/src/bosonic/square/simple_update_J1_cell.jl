"""
Simple Update for the bosonic square-lattice nearest-neighbour J1 model on a
periodic `Lx × Ly` iPEPS cell.

The tensor convention is `(L,D,R,U,p)`.  `lambda_x[x,y]` is the horizontal
bond on the left of `(x,y)`, and `lambda_y[x,y]` is the vertical bond below
`(x,y)`, exactly as in `simple_update_lib.jl`.

`Dmax` is the ordinary (state-counting) virtual dimension.  For the paper's
`D*=4, D=12` calculation use `Dmax=12`; the number and type of SU(2)
multiplets are selected by the SVD rather than prescribed here.
"""

Base.@kwdef struct SquareJ1SimpleUpdateSettings
    Dstar::Union{Nothing,Int} = 4
    Dmax::Int = 12
    multiplet_tol::Float64 = 1.0e-5
    convergence_tol::Float64 = 0.0
    print_every::Int = 1
    verbose::Bool = true
end

function _square_su_close(value_a, value_b, tolerance)
    tolerance > 0 || return false
    scale = max(abs(value_a), abs(value_b))
    iszero(scale) && return true
    return abs(value_a - value_b) <= tolerance * scale
end

"""
Keep at most `Dstar` reduced singular values globally across all SU(2)
sectors, subject also to the expanded-dimension safety cap `Dmax`.  A second
SVD with `truncspace` performs the actual TensorKit-space restriction.
"""
function square_su_tsvd_multiplets(tensor, Dstar::Int, Dmax::Int, multiplet_tol::Real)
    _, singular_values, _ = tsvd(tensor)
    singular_space = space(singular_values, 1)
    reduced_diagonal = diag(singular_values)
    candidates = [(
        value=abs(reduced_diagonal[sector][index]),
        sector=sector,
        index=index,
        quantum_dimension=dim(sector),
    ) for sector in sectors(singular_space)
      for index in eachindex(reduced_diagonal[sector])]
    sort!(candidates; by=entry -> entry.value, rev=true)

    keep_count = min(Dstar, length(candidates))
    while keep_count > 0 &&
          sum(entry.quantum_dimension for entry in @view(candidates[1:keep_count])) > Dmax
        keep_count -= 1
    end
    while 0 < keep_count < length(candidates) &&
          _square_su_close(
              candidates[keep_count].value,
              candidates[keep_count + 1].value,
              multiplet_tol,
          )
        boundary_value = candidates[keep_count].value
        keep_count -= 1
        while keep_count > 0 &&
              _square_su_close(candidates[keep_count].value, boundary_value, multiplet_tol)
            keep_count -= 1
        end
    end
    keep_count > 0 || error(
        "Dstar=$Dstar and Dmax=$Dmax leave no singular multiplet after truncation",
    )

    multiplicities = Dict(sector => 0 for sector in sectors(singular_space))
    for entry in @view(candidates[1:keep_count])
        multiplicities[entry.sector] += 1
    end
    kept_space = TensorKit.spacetype(singular_space)(
        sector => multiplicities[sector]
        for sector in sectors(singular_space)
        if multiplicities[sector] > 0
    )
    return tsvd(tensor; trunc=truncspace(kept_space))
end

function square_J1_initial_mixed_cell(
    cell_Lx::Int,
    cell_Ly::Int;
    Vp=SU2Space(1 / 2 => 1),
    Vv=SU2Space(0 => 1, 1 / 2 => 1),
)
    cell_Lx >= 2 || throw(ArgumentError("simple update currently requires Lx ≥ 2"))
    cell_Ly >= 2 || throw(ArgumentError("simple update currently requires Ly ≥ 2"))
    T_set, lambda_x, lambda_y = initial_iPEPS(cell_Lx, cell_Ly, Vp, Vv)
    for position in CartesianIndices(T_set)
        tensor_norm = norm(T_set[position])
        isfinite(tensor_norm) && !iszero(tensor_norm) || error(
            "empty SU(2) intertwiner space at site $position for Vv=$Vv",
        )
        T_set[position] /= tensor_norm
    end
    return T_set, lambda_x, lambda_y
end

_square_su_cycle_color(index::Int, length::Int) =
    isodd(length) && index == length ? 3 : (isodd(index) ? 1 : 2)

function _square_su_multiplet_count(V)
    return sum(dim(V, sector) for sector in sectors(V))
end

function _square_su_parity(V)
    sector_parities = unique(isodd(dim(sector)) ? :integer : :half_integer
                             for sector in sectors(V) if dim(V, sector) > 0)
    isempty(sector_parities) && return :empty
    length(sector_parities) == 1 && return only(sector_parities)
    return :mixed
end

function _square_su_space_record(lambda)
    V = space(lambda, 1)
    reduced_diagonal = diag(lambda)
    spectrum = [(
        sector=string(sector),
        values=collect(reduced_diagonal[sector]),
    ) for sector in sectors(V) if dim(V, sector) > 0]
    return (
        space=string(V),
        Dstar=_square_su_multiplet_count(V),
        D=dim(V),
        parity=_square_su_parity(V),
        lambda=spectrum,
    )
end

function _square_su_format_lambda(spectrum)
    return join(
        (entry.sector * "=>" * string(entry.values) for entry in spectrum),
        ", ",
    )
end

function square_J1_bond_space_report(lambda_x::AbstractMatrix, lambda_y::AbstractMatrix)
    size(lambda_x) == size(lambda_y) ||
        throw(DimensionMismatch("lambda_x and lambda_y must have the same cell size"))
    cell_Lx, cell_Ly = size(lambda_x)
    x_bonds = [(
        direction=:x,
        from=(cx, cy),
        to=(mod1(cx + 1, cell_Lx), cy),
        _square_su_space_record(lambda_x[mod1(cx + 1, cell_Lx), cy])...,
    ) for cy in 1:cell_Ly for cx in 1:cell_Lx]
    y_bonds = [(
        direction=:y,
        from=(cx, cy),
        to=(cx, mod1(cy + 1, cell_Ly)),
        _square_su_space_record(lambda_y[cx, mod1(cy + 1, cell_Ly)])...,
    ) for cx in 1:cell_Lx for cy in 1:cell_Ly]
    return vcat(x_bonds, y_bonds)
end

function square_J1_print_bond_spaces(lambda_x, lambda_y; prefix="")
    for bond in square_J1_bond_space_report(lambda_x, lambda_y)
        println(
            prefix,
            bond.direction,
            bond.from,
            "→",
            bond.to,
            ": ",
            bond.space,
            "  [D*=",
            bond.Dstar,
            ", D=",
            bond.D,
            ", parity=",
            bond.parity,
            "]  lambda={",
            _square_su_format_lambda(bond.lambda),
            "}",
        )
    end
    flush(stdout)
    return nothing
end

function _square_su_lambda_spectrum(lambda)
    return sort!(vec(abs.(diag(convert(Array, lambda)))); rev=true)
end

function _square_su_spectrum_distance(new_lambda, old_lambda)
    new_values = _square_su_lambda_spectrum(new_lambda)
    old_values = _square_su_lambda_spectrum(old_lambda)
    count = max(length(new_values), length(old_values))
    append!(new_values, zeros(eltype(new_values), count - length(new_values)))
    append!(old_values, zeros(eltype(old_values), count - length(old_values)))
    return norm(new_values - old_values)
end

function _square_su_convergence(lambda_x, lambda_y, old_x, old_y)
    errors = Float64[]
    for position in CartesianIndices(lambda_x)
        push!(errors, _square_su_spectrum_distance(lambda_x[position], old_x[position]))
        push!(errors, _square_su_spectrum_distance(lambda_y[position], old_y[position]))
    end
    return maximum(errors)
end

function _square_su_x_sweep!(
    step, T_set, lambda_x, lambda_y, gate, settings,
)
    cell_Lx, cell_Ly = size(T_set)
    for color in 1:(2 + Int(isodd(cell_Lx)))
        for cy in 1:cell_Ly, cx in 1:cell_Lx
            _square_su_cycle_color(cx, cell_Lx) == color || continue
            tebd_xbond(
                step, T_set, lambda_x, lambda_y, gate, cx + 0.5, cy,
                settings.Dmax;
                multiplet_tol=settings.multiplet_tol,
                Dstar=settings.Dstar,
                print_space=false,
            )
        end
    end
    return T_set, lambda_x, lambda_y
end


function _square_su_y_sweep!(
    step, T_set, lambda_x, lambda_y, gate, settings,
)
    cell_Lx, cell_Ly = size(T_set)
    for color in 1:(2 + Int(isodd(cell_Ly)))
        for cx in 1:cell_Lx, cy in 1:cell_Ly
            _square_su_cycle_color(cy, cell_Ly) == color || continue
            tebd_ybond(
                step, T_set, lambda_x, lambda_y, gate, cx, cy + 0.5,
                settings.Dmax;
                multiplet_tol=settings.multiplet_tol,
                Dstar=settings.Dstar,
                print_space=false,
            )
        end
    end
    return T_set, lambda_x, lambda_y
end

"""
    square_J1_simple_update_cell(T_set, lambda_x, lambda_y, tau, dt;
                                 J1=1, settings, callback=nothing)

Apply first-order imaginary-time Simple Update to every positive-x and
positive-y bond of an arbitrary periodic cell.  A callback, when supplied, is
called as `callback(T_set, lambda_x, lambda_y, step, report, error)` after each
complete x+y sweep.
"""
function square_J1_simple_update_cell(
    T_set::AbstractMatrix,
    lambda_x::AbstractMatrix,
    lambda_y::AbstractMatrix,
    tau::Real,
    dt::Real;
    J1::Real=1.0,
    settings::SquareJ1SimpleUpdateSettings=SquareJ1SimpleUpdateSettings(),
    callback=nothing,
)
    size(T_set) == size(lambda_x) == size(lambda_y) ||
        throw(DimensionMismatch("T_set, lambda_x and lambda_y must have the same size"))
    cell_Lx, cell_Ly = size(T_set)
    cell_Lx >= 2 || throw(ArgumentError("simple update currently requires Lx ≥ 2"))
    cell_Ly >= 2 || throw(ArgumentError("simple update currently requires Ly ≥ 2"))
    tau >= 0 || throw(ArgumentError("tau must be non-negative"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    settings.Dmax > 0 || throw(ArgumentError("Dmax must be positive"))
    isnothing(settings.Dstar) || settings.Dstar > 0 ||
        throw(ArgumentError("Dstar must be positive or nothing"))
    settings.multiplet_tol >= 0 ||
        throw(ArgumentError("multiplet_tol must be non-negative"))
    settings.print_every > 0 || throw(ArgumentError("print_every must be positive"))

    number_steps = Int(round(tau / dt))
    isapprox(number_steps * dt, tau; atol=100eps(Float64) * max(1, abs(tau))) ||
        throw(ArgumentError("tau/dt must be an integer; got tau=$tau and dt=$dt"))
    gate = prepare_gate_Heisenberg(dt * J1, "SU2")
    convergence_tol = iszero(settings.convergence_tol) ?
        abs(dt) * 1.0e-3 : settings.convergence_tol
    history = NamedTuple[]

    for step in 1:number_steps
        old_x = deepcopy(lambda_x)
        old_y = deepcopy(lambda_y)
        _square_su_x_sweep!(step, T_set, lambda_x, lambda_y, gate, settings)
        _square_su_y_sweep!(step, T_set, lambda_x, lambda_y, gate, settings)

        error = _square_su_convergence(lambda_x, lambda_y, old_x, old_y)
        report = square_J1_bond_space_report(lambda_x, lambda_y)
        push!(history, (step=step, tau=step * dt, error=error, bonds=report))
        if settings.verbose && (step == 1 || step % settings.print_every == 0)
            println("Simple Update step $step/$number_steps, tau=$(step * dt), convergence=$error")
            square_J1_print_bond_spaces(lambda_x, lambda_y; prefix="  ")
        end
        isnothing(callback) || callback(
            T_set, lambda_x, lambda_y, step, report, error,
        )
        error < convergence_tol && break
    end
    return T_set, lambda_x, lambda_y, history
end
