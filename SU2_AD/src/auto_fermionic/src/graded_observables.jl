function graded_build_MM_LU(CTM_cell::AbstractMatrix, AA_LU, cx::Integer, cy::Integer)
    Lx, Ly = size(CTM_cell)
    C1 = CTM_cell[mod1(cx, Lx), mod1(cy, Ly)].Cset.C1
    T1 = CTM_cell[mod1(cx + 1, Lx), mod1(cy, Ly)].Tset.T1
    T4 = CTM_cell[mod1(cx, Lx), mod1(cy + 1, Ly)].Tset.T4
    @tensor MM[:] := C1[1, 2] * T1[2, 3, -3] * T4[-1, 4, 1] * AA_LU[4, -2, -4, 3]
    return permute(MM, ((1, 2), (3, 4)))
end

function graded_build_MM_RU(CTM_cell::AbstractMatrix, AA_RU, cx::Integer, cy::Integer)
    Lx, Ly = size(CTM_cell)
    T1 = CTM_cell[mod1(cx + 2, Lx), mod1(cy, Ly)].Tset.T1
    C2 = CTM_cell[mod1(cx + 3, Lx), mod1(cy, Ly)].Cset.C2
    T2 = CTM_cell[mod1(cx + 3, Lx), mod1(cy + 1, Ly)].Tset.T2
    @tensor MM[:] := T1[-1, 3, 1] * C2[1, 2] * AA_RU[-2, -4, 4, 3] * T2[2, 4, -3]
    return permute(MM, ((1, 2), (3, 4)))
end

function graded_build_MM_LD(CTM_cell::AbstractMatrix, AA_LD, cx::Integer, cy::Integer)
    Lx, Ly = size(CTM_cell)
    T4 = CTM_cell[mod1(cx, Lx), mod1(cy + 2, Ly)].Tset.T4
    C4 = CTM_cell[mod1(cx, Lx), mod1(cy + 3, Ly)].Cset.C4
    T3 = CTM_cell[mod1(cx + 1, Lx), mod1(cy + 3, Ly)].Tset.T3
    @tensor MM[:] := T4[1, 3, -1] * AA_LD[3, 4, -4, -2] * C4[2, 1] * T3[-3, 4, 2]
    return permute(MM, ((1, 2), (3, 4)))
end

function graded_build_MM_RD(CTM_cell::AbstractMatrix, AA_RD, cx::Integer, cy::Integer)
    Lx, Ly = size(CTM_cell)
    T2 = CTM_cell[mod1(cx + 3, Lx), mod1(cy + 2, Ly)].Tset.T2
    T3 = CTM_cell[mod1(cx + 2, Lx), mod1(cy + 3, Ly)].Tset.T3
    C3 = CTM_cell[mod1(cx + 3, Lx), mod1(cy + 3, Ly)].Cset.C3
    @tensor MM[:] := T2[-4, -3, 2] * T3[1, -2, -1] * C3[2, 1]
    @tensor MM[:] := MM[-1, 1, 2, -3] * AA_RD[-2, 1, 2, -4]
    return permute(MM, ((1, 2), (3, 4)))
end

function graded_build_open_MM_LU(CTM_cell::AbstractMatrix, OAA_LU, cx::Integer, cy::Integer)
    Lx, Ly = size(CTM_cell)
    C1 = CTM_cell[mod1(cx, Lx), mod1(cy, Ly)].Cset.C1
    T1 = CTM_cell[mod1(cx + 1, Lx), mod1(cy, Ly)].Tset.T1
    T4 = CTM_cell[mod1(cx, Lx), mod1(cy + 1, Ly)].Tset.T4
    @tensor MM[:] := C1[1, 2] * T1[2, 3, -3] * T4[-1, 4, 1] * OAA_LU[4, -2, -4, 3, -5, -6]
    return permute(MM, ((1, 2, 5), (3, 4, 6)))
end

function graded_build_open_MM_RU(CTM_cell::AbstractMatrix, OAA_RU, cx::Integer, cy::Integer)
    Lx, Ly = size(CTM_cell)
    T1 = CTM_cell[mod1(cx + 2, Lx), mod1(cy, Ly)].Tset.T1
    C2 = CTM_cell[mod1(cx + 3, Lx), mod1(cy, Ly)].Cset.C2
    T2 = CTM_cell[mod1(cx + 3, Lx), mod1(cy + 1, Ly)].Tset.T2
    @tensor MM[:] := T1[-1, 3, 1] * C2[1, 2] * OAA_RU[-2, -4, 4, 3, -5, -6] * T2[2, 4, -3]
    return permute(MM, ((1, 2, 5), (3, 4, 6)))
end

function graded_build_open_MM_LD(CTM_cell::AbstractMatrix, OAA_LD, cx::Integer, cy::Integer)
    Lx, Ly = size(CTM_cell)
    T4 = CTM_cell[mod1(cx, Lx), mod1(cy + 2, Ly)].Tset.T4
    C4 = CTM_cell[mod1(cx, Lx), mod1(cy + 3, Ly)].Cset.C4
    T3 = CTM_cell[mod1(cx + 1, Lx), mod1(cy + 3, Ly)].Tset.T3
    @tensor MM[:] := T4[1, 3, -1] * OAA_LD[3, 4, -4, -2, -5, -6] * C4[2, 1] * T3[-3, 4, 2]
    return permute(MM, ((1, 2, 5), (3, 4, 6)))
end

function graded_build_open_MM_RD(CTM_cell::AbstractMatrix, OAA_RD, cx::Integer, cy::Integer)
    Lx, Ly = size(CTM_cell)
    T2 = CTM_cell[mod1(cx + 3, Lx), mod1(cy + 2, Ly)].Tset.T2
    T3 = CTM_cell[mod1(cx + 2, Lx), mod1(cy + 3, Ly)].Tset.T3
    C3 = CTM_cell[mod1(cx + 3, Lx), mod1(cy + 3, Ly)].Cset.C3
    @tensor MM[:] := T2[-4, -3, 2] * T3[1, -2, -1] * C3[2, 1]
    @tensor MM[:] := MM[-1, 1, 2, -3] * OAA_RD[-2, 1, 2, -4, -5, -6]
    return permute(MM, ((1, 2, 5), (3, 4, 6)))
end

function graded_ob_2x2(CTM_cell::AbstractMatrix, AA_LU, AA_RU, AA_LD, AA_RD, cx::Integer, cy::Integer)
    MM_LU = graded_build_MM_LU(CTM_cell, AA_LU, cx, cy)
    MM_RU = graded_build_MM_RU(CTM_cell, AA_RU, cx, cy)
    MM_LD = graded_build_MM_LD(CTM_cell, AA_LD, cx, cy)
    MM_RD = graded_build_MM_RD(CTM_cell, AA_RD, cx, cy)

    up = MM_LU * MM_RU
    down = MM_LD * MM_RD
    rho = @tensor up[1, 2, 3, 4] * down[1, 2, 3, 4]
    return rho
end

const _GRADED_PATCH_SITES = (:LU, :RU, :LD, :RD)

function _graded_patch_positions(AA_cell::AbstractMatrix, cx::Integer, cy::Integer)
    Lx, Ly = size(AA_cell)
    return (
        LU = (mod1(cx + 1, Lx), mod1(cy + 1, Ly)),
        RU = (mod1(cx + 2, Lx), mod1(cy + 1, Ly)),
        LD = (mod1(cx + 1, Lx), mod1(cy + 2, Ly)),
        RD = (mod1(cx + 2, Lx), mod1(cy + 2, Ly)),
    )
end

function _graded_patch_double_layers(AA_cell::AbstractMatrix, positions)
    return (
        LU = AA_cell[positions.LU...],
        RU = AA_cell[positions.RU...],
        LD = AA_cell[positions.LD...],
        RD = AA_cell[positions.RD...],
    )
end

function _graded_ob_2x2_patch(CTM_cell::AbstractMatrix, patch, cx::Integer, cy::Integer)
    return graded_ob_2x2(CTM_cell, patch.LU, patch.RU, patch.LD, patch.RD, cx, cy)
end

function _graded_patch_replace(patch, site::Symbol, AA_site)
    site === :LU && return (LU = AA_site, RU = patch.RU, LD = patch.LD, RD = patch.RD)
    site === :RU && return (LU = patch.LU, RU = AA_site, LD = patch.LD, RD = patch.RD)
    site === :LD && return (LU = patch.LU, RU = patch.RU, LD = AA_site, RD = patch.RD)
    site === :RD && return (LU = patch.LU, RU = patch.RU, LD = patch.LD, RD = AA_site)
    throw(ArgumentError("site must be one of $(_GRADED_PATCH_SITES), got $(repr(site))"))
end

function graded_ob_2x2_norm(CTM_cell::AbstractMatrix, AA_cell::AbstractMatrix, cx::Integer, cy::Integer)
    positions = _graded_patch_positions(AA_cell, cx, cy)
    patch = _graded_patch_double_layers(AA_cell, positions)
    return _graded_ob_2x2_patch(CTM_cell, patch, cx, cy)
end

function graded_ob_onsite_at(
    CTM_cell::AbstractMatrix,
    O,
    A_cell::AbstractMatrix,
    AA_cell::AbstractMatrix,
    cx::Integer,
    cy::Integer;
    site::Symbol = :LU,
)
    positions = _graded_patch_positions(AA_cell, cx, cy)
    patch = _graded_patch_double_layers(AA_cell, positions)
    haskey(positions, site) || throw(ArgumentError("site must be one of $(_GRADED_PATCH_SITES), got $(repr(site))"))

    AA_site, _, _, _, _ = graded_build_double_layer_direct(A_cell[getproperty(positions, site)...], O)
    ob_patch = _graded_patch_replace(patch, site, AA_site)
    ob = _graded_ob_2x2_patch(CTM_cell, ob_patch, cx, cy)
    norm_patch = _graded_ob_2x2_patch(CTM_cell, patch, cx, cy)
    return ob / norm_patch
end

function graded_ob_product_2x2(
    CTM_cell::AbstractMatrix,
    operators::NamedTuple,
    A_cell::AbstractMatrix,
    AA_cell::AbstractMatrix,
    cx::Integer,
    cy::Integer,
)
    positions = _graded_patch_positions(AA_cell, cx, cy)
    patch = _graded_patch_double_layers(AA_cell, positions)

    ob_patch = patch
    for site in keys(operators)
        haskey(positions, site) || throw(ArgumentError("site must be one of $(_GRADED_PATCH_SITES), got $(repr(site))"))
        AA_site, _, _, _, _ = graded_build_double_layer_direct(A_cell[getproperty(positions, site)...], getproperty(operators, site))
        ob_patch = _graded_patch_replace(ob_patch, site, AA_site)
    end

    ob = _graded_ob_2x2_patch(CTM_cell, ob_patch, cx, cy)
    norm_patch = _graded_ob_2x2_patch(CTM_cell, patch, cx, cy)
    return ob / norm_patch
end

function _graded_ob_triangle_2x2_raw(
    CTM_cell::AbstractMatrix,
    O3,
    A_cell::AbstractMatrix,
    AA_cell::AbstractMatrix,
    cx::Integer,
    cy::Integer;
    orientation::Symbol = :up,
)
    positions = _graded_patch_positions(AA_cell, cx, cy)
    patch = _graded_patch_double_layers(AA_cell, positions)

    if orientation === :up
        OAA_LD, _, _, _, _ = graded_build_open_double_layer_direct(A_cell[positions.LD...])
        OAA_RD, _, _, _, _ = graded_build_open_double_layer_direct(A_cell[positions.RD...])
        OAA_RU, _, _, _, _ = graded_build_open_double_layer_direct(A_cell[positions.RU...])

        MM_LU = graded_build_MM_LU(CTM_cell, patch.LU, cx, cy)
        MM_RU = graded_build_open_MM_RU(CTM_cell, OAA_RU, cx, cy)
        MM_LD = graded_build_open_MM_LD(CTM_cell, OAA_LD, cx, cy)
        MM_RD = graded_build_open_MM_RD(CTM_cell, OAA_RD, cx, cy)

        ob = @tensor MM_LU[1, 2, 3, 4] *
                     MM_RU[3, 4, 9, 5, 6, 12] *
                     MM_LD[1, 2, 7, 13, 14, 10] *
                     MM_RD[13, 14, 8, 5, 6, 11] *
                     O3[7, 8, 9, 10, 11, 12]
    elseif orientation === :down
        OAA_LD, _, _, _, _ = graded_build_open_double_layer_direct(A_cell[positions.LD...])
        OAA_LU, _, _, _, _ = graded_build_open_double_layer_direct(A_cell[positions.LU...])
        OAA_RU, _, _, _, _ = graded_build_open_double_layer_direct(A_cell[positions.RU...])

        MM_LU = graded_build_open_MM_LU(CTM_cell, OAA_LU, cx, cy)
        MM_RU = graded_build_open_MM_RU(CTM_cell, OAA_RU, cx, cy)
        MM_LD = graded_build_open_MM_LD(CTM_cell, OAA_LD, cx, cy)
        MM_RD = graded_build_MM_RD(CTM_cell, patch.RD, cx, cy)

        ob = @tensor MM_LU[1, 2, 8, 3, 4, 11] *
                     MM_RU[3, 4, 9, 5, 6, 12] *
                     MM_LD[1, 2, 7, 13, 14, 10] *
                     MM_RD[13, 14, 5, 6] *
                     O3[7, 8, 9, 10, 11, 12]
    else
        throw(ArgumentError("orientation must be :up or :down, got $(repr(orientation))"))
    end

    return ob
end

function graded_ob_triangle_2x2(
    CTM_cell::AbstractMatrix,
    O3,
    A_cell::AbstractMatrix,
    AA_cell::AbstractMatrix,
    cx::Integer,
    cy::Integer;
    orientation::Symbol = :up,
)
    ob = _graded_ob_triangle_2x2_raw(CTM_cell, O3, A_cell, AA_cell, cx, cy; orientation)
    norm_op = TensorKit.id(codomain(O3))
    norm = _graded_ob_triangle_2x2_raw(CTM_cell, norm_op, A_cell, AA_cell, cx, cy; orientation)
    return ob / norm
end

function graded_ob_onsite(CTM_cell::AbstractMatrix, O, A_cell::AbstractMatrix, AA_cell::AbstractMatrix, cx::Integer, cy::Integer)
    return graded_ob_onsite_at(CTM_cell, O, A_cell, AA_cell, cx, cy; site = :LU)
end

function graded_ob_onsite_cell(CTM_cell::AbstractMatrix, O, A_cell::AbstractMatrix, AA_cell::AbstractMatrix; site::Symbol = :LU)
    Lx, Ly = size(AA_cell)
    values = Matrix{ComplexF64}(undef, Lx, Ly)
    for cx in 1:Lx, cy in 1:Ly
        values[cx, cy] = graded_ob_onsite_at(CTM_cell, O, A_cell, AA_cell, cx, cy; site)
    end
    return values
end

function graded_spinful_onsite_observables(CTM_cell::AbstractMatrix, A_cell::AbstractMatrix, AA_cell::AbstractMatrix)
    ops = tensorkit_spinful_operators()
    return (
        id = graded_ob_onsite_cell(CTM_cell, ops.id, A_cell, AA_cell),
        n_total = graded_ob_onsite_cell(CTM_cell, ops.n_total, A_cell, AA_cell),
        n_up = graded_ob_onsite_cell(CTM_cell, ops.n_up, A_cell, AA_cell),
        n_dn = graded_ob_onsite_cell(CTM_cell, ops.n_dn, A_cell, AA_cell),
        n_double = graded_ob_onsite_cell(CTM_cell, ops.n_double, A_cell, AA_cell),
        sx = graded_ob_onsite_cell(CTM_cell, ops.sx, A_cell, AA_cell),
        sy = graded_ob_onsite_cell(CTM_cell, ops.sy, A_cell, AA_cell),
        sz = graded_ob_onsite_cell(CTM_cell, ops.sz, A_cell, AA_cell),
    )
end
