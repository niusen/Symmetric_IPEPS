const SQUARE_J1_PAPER_VEVEN = SU2Space(0 => 1, 1 => 2, 2 => 1)
const SQUARE_J1_PAPER_VODD = SU2Space(1 / 2 => 2, 3 / 2 => 2)

# Smallest parity-resolved spaces that can seed the paper's 2x2 matching.
# Every site must touch exactly one odd bond so that its four virtual legs can
# fuse to the physical spin-1/2 representation.
const SQUARE_J1_MINIMAL_VEVEN = SU2Space(0 => 1, 1 => 1)
const SQUARE_J1_MINIMAL_VODD = SU2Space(1 / 2 => 1)

function square_J1_diagonal_identity(V)
    reduced_dimension = sum(dim(V, sector) for sector in sectors(V))
    return DiagonalTensorMap(ones(Float64, reduced_dimension), V)
end

function square_J1_random_site_tensor(VL, VD, VR, VU; Vp=SU2Space(1 / 2 => 1))
    A = TensorMap(randn, VL ⊗ VD ⊗ VR' ⊗ VU', Vp)
    A = permute(A, (1, 2, 3, 4, 5), ())
    norm_A = norm(A)
    isfinite(norm_A) && norm_A > 0 || error(
        "empty intertwiner for (VL,VD,VR,VU)=($VL,$VD,$VR,$VU)",
    )
    return A / norm_A
end

function square_J1_cell_from_bond_spaces(
    xleft::AbstractMatrix,
    ybelow::AbstractMatrix,
)
    size(xleft) == size(ybelow) ||
        throw(DimensionMismatch("xleft and ybelow must have the same size"))
    cell_Lx, cell_Ly = size(xleft)
    T_set = Matrix{Any}(undef, cell_Lx, cell_Ly)
    lambda_x = Matrix{Any}(undef, cell_Lx, cell_Ly)
    lambda_y = Matrix{Any}(undef, cell_Lx, cell_Ly)
    for cx in 1:cell_Lx, cy in 1:cell_Ly
        VL = xleft[cx, cy]
        VD = ybelow[cx, cy]
        VR = xleft[mod1(cx + 1, cell_Lx), cy]
        VU = ybelow[cx, mod1(cy + 1, cell_Ly)]
        T_set[cx, cy] = square_J1_random_site_tensor(VL, VD, VR, VU)
        lambda_x[cx, cy] = square_J1_diagonal_identity(VL)
        lambda_y[cx, cy] = square_J1_diagonal_identity(VD')
    end
    return T_set, lambda_x, lambda_y
end

function square_J1_homogeneous_cell(V, cell_Lx::Int=2, cell_Ly::Int=2)
    spaces = fill(V, cell_Lx, cell_Ly)
    return square_J1_cell_from_bond_spaces(spaces, spaces)
end

function square_J1_matching_cell(kind::Symbol, Veven, Vodd)
    xleft = fill(Veven, 2, 2)
    ybelow = fill(Veven, 2, 2)
    if kind === :x_columnar
        xleft[2, 1] = Vodd
        xleft[2, 2] = Vodd
    elseif kind === :x_staggered
        xleft[2, 1] = Vodd
        xleft[1, 2] = Vodd
    elseif kind === :y_columnar
        ybelow[1, 2] = Vodd
        ybelow[2, 2] = Vodd
    elseif kind === :y_staggered
        ybelow[1, 2] = Vodd
        ybelow[2, 1] = Vodd
    else
        throw(ArgumentError("unknown 2x2 matching $kind"))
    end
    return square_J1_cell_from_bond_spaces(xleft, ybelow)
end

function square_J1_named_initial_state(kind::Symbol, seed::Int=666)
    Random.seed!(seed)
    if kind === :mixed_min
        return square_J1_homogeneous_cell(SU2Space(0 => 1, 1 / 2 => 1))
    elseif kind === :mixed_balanced
        return square_J1_homogeneous_cell(SU2Space(0 => 2, 1 / 2 => 2))
    elseif kind === :mixed_broad
        return square_J1_homogeneous_cell(
            SU2Space(0 => 1, 1 / 2 => 1, 1 => 1, 3 / 2 => 1),
        )
    elseif kind === :paper_union
        return square_J1_homogeneous_cell(
            SU2Space(0 => 1, 1 / 2 => 2, 1 => 2, 3 / 2 => 2, 2 => 1),
        )
    end

    name = string(kind)
    for (prefix, Veven, Vodd) in (
        ("paper_", SQUARE_J1_PAPER_VEVEN, SQUARE_J1_PAPER_VODD),
        ("minimal_", SQUARE_J1_MINIMAL_VEVEN, SQUARE_J1_MINIMAL_VODD),
    )
        startswith(name, prefix) || continue
        matching = Symbol(name[(length(prefix) + 1):end])
        return square_J1_matching_cell(matching, Veven, Vodd)
    end
    throw(ArgumentError("unknown square-J1 initialization $kind"))
end
