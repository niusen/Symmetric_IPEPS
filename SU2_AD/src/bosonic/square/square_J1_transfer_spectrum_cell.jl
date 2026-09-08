"""
Transfer-matrix spectrum adapter for a square-lattice unit cell.

The tensor contraction itself is not reimplemented here.  Every site step
calls the established three-line `correl_TransOp` from `square_correl.jl`,
which contracts the upper CTM edge, closed iPEPS double layer, and lower CTM
edge in its original order.  This file only repeats that existing site step
over one full unit-cell period and organizes the SU(2)-sector eigensolves.
"""

function _square_J1_transfer_cell_get(cell, cx::Int, cy::Int, Lx::Int, Ly::Int)
    x, y = mod1(cx, Lx), mod1(cy, Ly)
    return cell isa AbstractMatrix ? cell[x, y] : cell[x][y]
end

function _square_J1_apply_existing_three_line_transfer(
    vector,
    environment,
    direction::Symbol,
    cut::Int,
    Lx::Int,
    Ly::Int,
)
    result = vector
    if direction === :x
        cy = mod1(cut, Ly)
        for cx in 1:Lx
            top = _square_J1_transfer_cell_get(
                environment.CTM.Tset, cx, cy - 1, Lx, Ly,
            ).T1
            AA = _square_J1_transfer_cell_get(environment.AA, cx, cy, Lx, Ly)
            bottom = _square_J1_transfer_cell_get(
                environment.CTM.Tset, cx, cy + 1, Lx, Ly,
            ).T3
            result = correl_TransOp(result, top, bottom, AA, true)
        end
    elseif direction === :y
        cx = mod1(cut, Lx)
        for cy in 1:Ly
            right = _square_J1_transfer_cell_get(
                environment.CTM.Tset, cx + 1, cy, Lx, Ly,
            ).T2
            AA = _square_J1_transfer_cell_get(environment.AA, cx, cy, Lx, Ly)
            AA_rotated = permute(AA, (4, 1, 2, 3), ())
            left = _square_J1_transfer_cell_get(
                environment.CTM.Tset, cx - 1, cy, Lx, Ly,
            ).T4
            result = correl_TransOp(result, right, left, AA_rotated, true)
        end
    else
        error("direction must be :x or :y")
    end
    return result
end

function square_J1_transfer_spectrum_three_line_cell(
    A_set,
    environment,
    direction::Symbol;
    n_values::Int=6,
    spins=(0, 1 / 2, 1, 3 / 2, 2),
    cut::Int=1,
)
    Lx, Ly = size(A_set)
    if direction === :x
        cy = mod1(cut, Ly)
        top = _square_J1_transfer_cell_get(environment.CTM.Tset, 1, cy - 1, Lx, Ly).T1
        AA = _square_J1_transfer_cell_get(environment.AA, 1, cy, Lx, Ly)
        bottom = _square_J1_transfer_cell_get(environment.CTM.Tset, 1, cy + 1, Lx, Ly).T3
    elseif direction === :y
        cx = mod1(cut, Lx)
        top = _square_J1_transfer_cell_get(environment.CTM.Tset, cx + 1, 1, Lx, Ly).T2
        AA = permute(
            _square_J1_transfer_cell_get(environment.AA, cx, 1, Lx, Ly),
            (4, 1, 2, 3),
            (),
        )
        bottom = _square_J1_transfer_cell_get(environment.CTM.Tset, cx - 1, 1, Lx, Ly).T4
    else
        error("direction must be :x or :y")
    end

    action(vector) = _square_J1_apply_existing_three_line_transfer(
        vector, environment, direction, cut, Lx, Ly,
    )
    eigenvalues = ComplexF64[]
    spin_labels = Float64[]
    failed_sectors = String[]
    for spin in spins
        initial = permute(
            TensorMap(
                randn,
                SU2Space(spin => 1) ⊗ space(top, 1)' ⊗ space(AA, 1)',
                space(bottom, 3),
            ),
            (1, 2, 3, 4),
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
    isempty(eigenvalues) && error("no three-line transfer-matrix eigenvalue was obtained")

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
        geometry="three-line T-AA-T",
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
