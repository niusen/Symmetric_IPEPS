struct GradedCset{TC1,TC2,TC3,TC4}
    C1::TC1
    C2::TC2
    C3::TC3
    C4::TC4
end

struct GradedTset{TT1,TT2,TT3,TT4}
    T1::TT1
    T2::TT2
    T3::TT3
    T4::TT4
end

struct GradedCTM{TC,TT}
    Cset::TC
    Tset::TT
end

function _normalize_tensormap(A)
    nrm = norm(A)
    return nrm == 0 ? A : A / nrm
end

function _diagonal_inv_sqrt(S::DiagonalTensorMap; cutoff::Real=0)
    R = deepcopy(S)
    if sectortype(S) == Trivial
        vals = S.data
        R.data .= ifelse.(abs.(vals) .> cutoff, vals .^ (-1 / 2), zero(eltype(vals)))
    else
        for (sector, vals) in blocks(S)
            block(R, sector) .= ifelse.(abs.(vals) .> cutoff, vals .^ (-1 / 2), zero(eltype(vals)))
        end
    end
    return R
end

_ctm_index(i::Integer) = mod1(i, 4)

function _get_cset(C::GradedCset, i::Integer)
    j = _ctm_index(i)
    return j == 1 ? C.C1 : j == 2 ? C.C2 : j == 3 ? C.C3 : C.C4
end

function _get_tset(T::GradedTset, i::Integer)
    j = _ctm_index(i)
    return j == 1 ? T.T1 : j == 2 ? T.T2 : j == 3 ? T.T3 : T.T4
end

function _set_cset(C::GradedCset, i::Integer, value)
    j = _ctm_index(i)
    return GradedCset(
        j == 1 ? value : C.C1,
        j == 2 ? value : C.C2,
        j == 3 ? value : C.C3,
        j == 4 ? value : C.C4,
    )
end

function _set_tset(T::GradedTset, i::Integer, value)
    j = _ctm_index(i)
    return GradedTset(
        j == 1 ? value : T.T1,
        j == 2 ? value : T.T2,
        j == 3 ? value : T.T3,
        j == 4 ? value : T.T4,
    )
end

function graded_rotate_double_layer(AA, direction::Integer)
    direction in 1:4 || error("direction must be 1, 2, 3, or 4")
    order = (
        mod1(2 - direction, 4),
        mod1(3 - direction, 4),
        mod1(4 - direction, 4),
        mod1(1 - direction, 4),
    )
    return permute(AA, order)
end

function _as_indexable_cell(A_cell::AbstractMatrix)
    return A_cell
end

function _cell_size(A_cell)
    return size(A_cell, 1), size(A_cell, 2)
end

function graded_double_layer_cell(A_cell::AbstractMatrix; builder::Symbol=:direct)
    Lx, Ly = _cell_size(A_cell)
    AA_cell = Matrix{Any}(undef, Lx, Ly)
    UL_cell = Matrix{Any}(undef, Lx, Ly)
    UD_cell = Matrix{Any}(undef, Lx, Ly)
    UR_cell = Matrix{Any}(undef, Lx, Ly)
    UU_cell = Matrix{Any}(undef, Lx, Ly)

    build = builder === :direct ? graded_build_double_layer_direct :
            builder === :legacy_path ? graded_build_double_layer :
            error("unknown graded double-layer builder: $builder")

    for cx in 1:Lx, cy in 1:Ly
        AA, U_L, U_D, U_R, U_U = build(A_cell[cx, cy])
        AA_cell[cx, cy] = AA
        UL_cell[cx, cy] = U_L
        UD_cell[cx, cy] = U_D
        UR_cell[cx, cy] = U_R
        UU_cell[cx, cy] = U_U
    end

    return (AA=AA_cell, U_L=UL_cell, U_D=UD_cell, U_R=UR_cell, U_U=UU_cell)
end

function _randn_tensormap(cod, dom)
    return randn(ComplexF64, cod, dom)
end

function graded_init_ctm(AA_fused, U_L, U_D, U_R, U_U; init_type::String="PBC")
    if init_type == "PBC"
        @tensor C1[:] := AA_fused[1, -1, -2, 3] * U_L'[2, 2, 1] * U_U'[3, 4, 4]
        @tensor C2[:] := AA_fused[-1, -2, 3, 1] * U_U'[1, 2, 2] * U_R'[3, 4, 4]
        @tensor C3[:] := AA_fused[-2, 3, 1, -1] * U_R'[1, 2, 2] * U_D'[4, 4, 3]
        @tensor C4[:] := AA_fused[1, 3, -1, -2] * U_L'[2, 2, 1] * U_D'[4, 4, 3]

        @tensor T4[:] := AA_fused[1, -1, -2, -3] * U_L'[2, 2, 1]
        @tensor T1[:] := AA_fused[-1, -2, -3, 1] * U_U'[1, 2, 2]
        @tensor T2[:] := AA_fused[-2, -3, 1, -1] * U_R'[1, 2, 2]
        @tensor T3[:] := AA_fused[-3, 1, -1, -2] * U_D'[2, 2, 1]
    elseif init_type == "random"
        C1 = _randn_tensormap(space(AA_fused, 2), space(AA_fused, 3)')
        C2 = _randn_tensormap(space(AA_fused, 1), space(AA_fused, 2)')
        C3 = _randn_tensormap(space(AA_fused, 4), space(AA_fused, 1)')
        C4 = _randn_tensormap(space(AA_fused, 3), space(AA_fused, 4)')

        T4 = _randn_tensormap(space(AA_fused, 2) * space(AA_fused, 3), space(AA_fused, 4)')
        T1 = _randn_tensormap(space(AA_fused, 1) * space(AA_fused, 2), space(AA_fused, 3)')
        T2 = _randn_tensormap(space(AA_fused, 4) * space(AA_fused, 1), space(AA_fused, 2)')
        T3 = _randn_tensormap(space(AA_fused, 3) * space(AA_fused, 4), space(AA_fused, 1)')
    else
        error("unknown CTM init_type: $init_type")
    end

    return GradedCTM(GradedCset(C1, C2, C3, C4), GradedTset(T1, T2, T3, T4))
end

function graded_init_ctm_cell(A_cell::AbstractMatrix; init_type::String="PBC", builder::Symbol=:direct)
    dl = graded_double_layer_cell(A_cell; builder)
    Lx, Ly = size(A_cell)
    ctm_cell = Matrix{Any}(undef, Lx, Ly)
    for cx in 1:Lx, cy in 1:Ly
        ctm_cell[cx, cy] = graded_init_ctm(
            dl.AA[cx, cy],
            dl.U_L[cx, cy],
            dl.U_D[cx, cy],
            dl.U_R[cx, cy],
            dl.U_U[cx, cy];
            init_type,
        )
    end
    return (double_layer=dl, CTM=ctm_cell)
end

function graded_corner_transfer_matrices(CTM::GradedCTM, AA; direction::Integer=1)
    AArot = graded_rotate_double_layer(AA, direction)
    C1 = _get_cset(CTM.Cset, direction)
    C2 = _get_cset(CTM.Cset, direction + 1)
    C3 = _get_cset(CTM.Cset, direction - 2)
    C4 = _get_cset(CTM.Cset, direction - 1)
    T1 = _get_tset(CTM.Tset, direction)
    T2 = _get_tset(CTM.Tset, direction + 1)
    T3 = _get_tset(CTM.Tset, direction - 2)
    T4 = _get_tset(CTM.Tset, direction - 1)

    @tensor MMup[:] := C1[1, 2] * T1[2, 3, -3] * T4[-1, 4, 1] * AArot[4, -2, -4, 3]
    @tensor MMlow[:] := T4[1, 3, -1] * AArot[3, 4, -4, -2] * C4[2, 1] * T3[-3, 4, 2]
    @tensor MMup_reflect[:] := T1[-1, 3, 1] * C2[1, 2] * AArot[-2, -4, 4, 3] * T2[2, 4, -3]
    @tensor MMlow_reflect[:] := T2[-4, -3, 2] * T3[1, -2, -1] * C3[2, 1]
    @tensor MMlow_reflect[:] := MMlow_reflect[-1, 1, 2, -3] * AArot[-2, 1, 2, -4]

    return (
        MMup=permute(MMup, ((1, 2), (3, 4))),
        MMlow=permute(MMlow, ((1, 2), (3, 4))),
        MMup_reflect=permute(MMup_reflect, ((1, 2), (3, 4))),
        MMlow_reflect=permute(MMlow_reflect, ((1, 2), (3, 4))),
    )
end

function graded_ctm_projectors(CTM::GradedCTM, AA; direction::Integer=1, chi::Union{Nothing,Int}=nothing, cutoff::Real=1e-14)
    corners = graded_corner_transfer_matrices(CTM, AA; direction)

    RMup = permute(corners.MMup * corners.MMup_reflect, ((3, 4), (1, 2)))
    RMlow = corners.MMlow * corners.MMlow_reflect
    RMup = _normalize_tensormap(RMup)
    RMlow = _normalize_tensormap(RMlow)

    M = RMup * RMlow
    U, S, V = if chi === nothing
        svd_compact(M)
    else
        Utr, Str, Vtr, _ = svd_trunc(M; trunc=truncrank(chi))
        (Utr, Str, Vtr)
    end
    S = _normalize_tensormap(S)
    Sinvsqrt = _diagonal_inv_sqrt(S; cutoff)

    Pinv = RMlow * V' * Sinvsqrt
    P = Sinvsqrt * U' * RMup
    P = permute(P, ((2, 3), (1,)))

    return (P=P, Pinv=Pinv, singular_values=S, RMup=RMup, RMlow=RMlow)
end

function graded_ctm_directional_update(CTM::GradedCTM, AA; direction::Integer=1, chi::Union{Nothing,Int}=nothing, cutoff::Real=1e-14)
    direction in 1:4 || error("direction must be 1, 2, 3, or 4")
    AArot = graded_rotate_double_layer(AA, direction)
    projectors = graded_ctm_projectors(CTM, AA; direction, chi, cutoff)
    P = projectors.P
    Pinv = projectors.Pinv

    C1 = _get_cset(CTM.Cset, direction)
    C4 = _get_cset(CTM.Cset, direction - 1)
    T1 = _get_tset(CTM.Tset, direction)
    T3 = _get_tset(CTM.Tset, direction - 2)
    T4 = _get_tset(CTM.Tset, direction - 1)

    @tensor T4new[:] := T4[4, 3, 1] * AArot[3, 5, -2, 2] * Pinv[4, 5, -1] * P[1, 2, -3]
    @tensor C1new[:] := C1[1, 2] * T1[2, 3, -2] * Pinv[1, 3, -1]
    @tensor C4new[:] := C4[1, 2] * T3[-1, 3, 1] * P[2, 3, -2]

    Cset = _set_cset(CTM.Cset, direction, _normalize_tensormap(C1new))
    Cset = _set_cset(Cset, direction - 1, _normalize_tensormap(C4new))
    Tset = _set_tset(CTM.Tset, direction - 1, _normalize_tensormap(T4new))
    return (CTM=GradedCTM(Cset, Tset), projectors=projectors)
end

function graded_ctm_left_update(CTM::GradedCTM, AA; chi::Union{Nothing,Int}=nothing, cutoff::Real=1e-14)
    return graded_ctm_directional_update(CTM, AA; direction=1, chi, cutoff)
end

function graded_ctm_single_site_sweep(CTM::GradedCTM, AA; direction_order=(3, 4, 1, 2), chi::Union{Nothing,Int}=nothing, cutoff::Real=1e-14)
    current = CTM
    projectors = Vector{Any}(undef, length(direction_order))
    for (n, direction) in pairs(direction_order)
        result = graded_ctm_directional_update(current, AA; direction, chi, cutoff)
        current = result.CTM
        projectors[n] = result.projectors
    end
    return (CTM=current, projectors=projectors)
end

function _cell_pos(cx::Integer, cy::Integer, dx::Integer, dy::Integer, direction::Integer, Lx::Integer, Ly::Integer)
    if direction == 1
        return mod1(cx + dx, Lx), mod1(cy + dy, Ly)
    elseif direction == 2
        return mod1(cy - dy, Lx), mod1(cx + dx, Ly)
    elseif direction == 3
        return mod1(cx - dx, Lx), mod1(cy - dy, Ly)
    elseif direction == 4
        return mod1(cy + dy, Lx), mod1(cx - dx, Ly)
    else
        error("direction must be 1, 2, 3, or 4")
    end
end

function _rotated_AA_at(AA_cell::AbstractMatrix, cx::Integer, cy::Integer, direction::Integer)
    return graded_rotate_double_layer(AA_cell[cx, cy], direction)
end

function _graded_ctm_cell_projectors(CTM_cell::AbstractMatrix, AA_cell::AbstractMatrix, cx::Integer, cy::Integer, direction::Integer; chi::Union{Nothing,Int}=nothing, cutoff::Real=1e-14)
    Lx, Ly = size(AA_cell)

    ax, ay = _cell_pos(cx, cy, 1, 1, direction, Lx, Ly)
    AA = _rotated_AA_at(AA_cell, ax, ay, direction)
    c1x, c1y = _cell_pos(cx, cy, 0, 0, direction, Lx, Ly)
    t1x, t1y = _cell_pos(cx, cy, 1, 0, direction, Lx, Ly)
    t4x, t4y = _cell_pos(cx, cy, 0, 1, direction, Lx, Ly)
    C1 = _get_cset(CTM_cell[c1x, c1y].Cset, direction)
    T1 = _get_tset(CTM_cell[t1x, t1y].Tset, direction)
    T4 = _get_tset(CTM_cell[t4x, t4y].Tset, direction - 1)
    @tensor MMup[:] := C1[1, 2] * T1[2, 3, -3] * T4[-1, 4, 1] * AA[4, -2, -4, 3]

    ax, ay = _cell_pos(cx, cy, 1, 2, direction, Lx, Ly)
    AA = _rotated_AA_at(AA_cell, ax, ay, direction)
    t4x, t4y = _cell_pos(cx, cy, 0, 2, direction, Lx, Ly)
    c4x, c4y = _cell_pos(cx, cy, 0, 3, direction, Lx, Ly)
    t3x, t3y = _cell_pos(cx, cy, 1, 3, direction, Lx, Ly)
    T4 = _get_tset(CTM_cell[t4x, t4y].Tset, direction - 1)
    C4 = _get_cset(CTM_cell[c4x, c4y].Cset, direction - 1)
    T3 = _get_tset(CTM_cell[t3x, t3y].Tset, direction - 2)
    @tensor MMlow[:] := T4[1, 3, -1] * AA[3, 4, -4, -2] * C4[2, 1] * T3[-3, 4, 2]

    ax, ay = _cell_pos(cx, cy, 2, 1, direction, Lx, Ly)
    AA = _rotated_AA_at(AA_cell, ax, ay, direction)
    t1x, t1y = _cell_pos(cx, cy, 2, 0, direction, Lx, Ly)
    c2x, c2y = _cell_pos(cx, cy, 3, 0, direction, Lx, Ly)
    t2x, t2y = _cell_pos(cx, cy, 3, 1, direction, Lx, Ly)
    T1 = _get_tset(CTM_cell[t1x, t1y].Tset, direction)
    C2 = _get_cset(CTM_cell[c2x, c2y].Cset, direction + 1)
    T2 = _get_tset(CTM_cell[t2x, t2y].Tset, direction + 1)
    @tensor MMup_reflect[:] := T1[-1, 3, 1] * C2[1, 2] * AA[-2, -4, 4, 3] * T2[2, 4, -3]

    ax, ay = _cell_pos(cx, cy, 2, 2, direction, Lx, Ly)
    AA = _rotated_AA_at(AA_cell, ax, ay, direction)
    t2x, t2y = _cell_pos(cx, cy, 3, 2, direction, Lx, Ly)
    t3x, t3y = _cell_pos(cx, cy, 2, 3, direction, Lx, Ly)
    c3x, c3y = _cell_pos(cx, cy, 3, 3, direction, Lx, Ly)
    T2 = _get_tset(CTM_cell[t2x, t2y].Tset, direction + 1)
    T3 = _get_tset(CTM_cell[t3x, t3y].Tset, direction - 2)
    C3 = _get_cset(CTM_cell[c3x, c3y].Cset, direction - 2)
    @tensor MMlow_reflect[:] := T2[-4, -3, 2] * T3[1, -2, -1] * C3[2, 1]
    @tensor MMlow_reflect[:] := MMlow_reflect[-1, 1, 2, -3] * AA[-2, 1, 2, -4]

    MMup = permute(MMup, ((1, 2), (3, 4)))
    MMlow = permute(MMlow, ((1, 2), (3, 4)))
    MMup_reflect = permute(MMup_reflect, ((1, 2), (3, 4)))
    MMlow_reflect = permute(MMlow_reflect, ((1, 2), (3, 4)))

    RMup = permute(MMup * MMup_reflect, ((3, 4), (1, 2)))
    RMlow = MMlow * MMlow_reflect
    RMup = _normalize_tensormap(RMup)
    RMlow = _normalize_tensormap(RMlow)
    M = RMup * RMlow

    U, S, V = if chi === nothing
        svd_compact(M)
    else
        Utr, Str, Vtr, _ = svd_trunc(M; trunc=truncrank(chi))
        (Utr, Str, Vtr)
    end
    S = _normalize_tensormap(S)
    Sinvsqrt = _diagonal_inv_sqrt(S; cutoff)
    Pinv = RMlow * V' * Sinvsqrt
    P = Sinvsqrt * U' * RMup
    P = permute(P, ((2, 3), (1,)))

    return (P=P, Pinv=Pinv, singular_values=S)
end

function graded_ctm_cell_directional_update(CTM_cell::AbstractMatrix, AA_cell::AbstractMatrix; direction::Integer=1, chi::Union{Nothing,Int}=nothing, cutoff::Real=1e-14)
    direction in 1:4 || error("direction must be 1, 2, 3, or 4")
    Lx, Ly = size(AA_cell)
    size(CTM_cell) == (Lx, Ly) || error("CTM_cell and AA_cell must have the same size")

    PM_cell = Matrix{Any}(undef, Lx, Ly)
    PMinv_cell = Matrix{Any}(undef, Lx, Ly)
    projectors = Matrix{Any}(undef, Lx, Ly)

    cx_max, cy_max = direction in (1, 3) ? (Lx, Ly) : (Ly, Lx)
    for cx in 1:cx_max, cy in 1:cy_max
        proj = _graded_ctm_cell_projectors(CTM_cell, AA_cell, cx, cy, direction; chi, cutoff)
        px, py = _cell_pos(cx, cy, 0, 2, direction, Lx, Ly)
        PM_cell[px, py] = proj.P
        px, py = _cell_pos(cx, cy, 0, 1, direction, Lx, Ly)
        PMinv_cell[px, py] = proj.Pinv
        projectors[mod1(cx, Lx), mod1(cy, Ly)] = proj
    end

    new_cell = copy(CTM_cell)
    for cx in 1:cx_max, cy in 1:cy_max
        ax, ay = _cell_pos(cx, cy, 1, 2, direction, Lx, Ly)
        AA = _rotated_AA_at(AA_cell, ax, ay, direction)
        t4x, t4y = _cell_pos(cx, cy, 0, 2, direction, Lx, Ly)
        t1x, t1y = _cell_pos(cx, cy, 1, 0, direction, Lx, Ly)
        t3x, t3y = _cell_pos(cx, cy, 1, 3, direction, Lx, Ly)
        c1x, c1y = _cell_pos(cx, cy, 0, 0, direction, Lx, Ly)
        c4x, c4y = _cell_pos(cx, cy, 0, 3, direction, Lx, Ly)

        T4 = _get_tset(CTM_cell[t4x, t4y].Tset, direction - 1)
        T1 = _get_tset(CTM_cell[t1x, t1y].Tset, direction)
        T3 = _get_tset(CTM_cell[t3x, t3y].Tset, direction - 2)
        C1 = _get_cset(CTM_cell[c1x, c1y].Cset, direction)
        C4 = _get_cset(CTM_cell[c4x, c4y].Cset, direction - 1)

        pxa, pya = _cell_pos(cx, cy, 0, 2, direction, Lx, Ly)
        P = PM_cell[pxa, pya]
        Pinv = PMinv_cell[pxa, pya]
        @tensor T4new[:] := T4[4, 3, 1] * AA[3, 5, -2, 2] * Pinv[4, 5, -1] * P[1, 2, -3]

        pxa, pya = _cell_pos(cx, cy, 0, 0, direction, Lx, Ly)
        Pinv = PMinv_cell[pxa, pya]
        @tensor C1new[:] := C1[1, 2] * T1[2, 3, -2] * Pinv[1, 3, -1]

        pxa, pya = _cell_pos(cx, cy, 0, 3, direction, Lx, Ly)
        P = PM_cell[pxa, pya]
        @tensor C4new[:] := C4[1, 2] * T3[-1, 3, 1] * P[2, 3, -2]

        sx, sy = _cell_pos(cx, cy, 1, 0, direction, Lx, Ly)
        old = new_cell[sx, sy]
        new_cell[sx, sy] = GradedCTM(_set_cset(old.Cset, direction, _normalize_tensormap(C1new)), old.Tset)

        sx, sy = _cell_pos(cx, cy, 1, 2, direction, Lx, Ly)
        old = new_cell[sx, sy]
        new_cell[sx, sy] = GradedCTM(old.Cset, _set_tset(old.Tset, direction - 1, _normalize_tensormap(T4new)))

        sx, sy = _cell_pos(cx, cy, 1, 3, direction, Lx, Ly)
        old = new_cell[sx, sy]
        new_cell[sx, sy] = GradedCTM(_set_cset(old.Cset, direction - 1, _normalize_tensormap(C4new)), old.Tset)
    end

    return (CTM=new_cell, projectors=projectors)
end

function graded_ctm_cell_sweep(CTM_cell::AbstractMatrix, AA_cell::AbstractMatrix; direction_order=(3, 4, 1, 2), chi::Union{Nothing,Int}=nothing, cutoff::Real=1e-14)
    current = CTM_cell
    projectors = Vector{Any}(undef, length(direction_order))
    for (n, direction) in pairs(direction_order)
        result = graded_ctm_cell_directional_update(current, AA_cell; direction, chi, cutoff)
        current = result.CTM
        projectors[n] = result.projectors
    end
    return (CTM=current, projectors=projectors)
end

function graded_ctm_norm_signature(CTM_cell::AbstractMatrix)
    signature = Float64[]
    for cx in axes(CTM_cell, 1), cy in axes(CTM_cell, 2)
        CTM = CTM_cell[cx, cy]
        append!(
            signature,
            Float64[
                norm(CTM.Cset.C1), norm(CTM.Cset.C2), norm(CTM.Cset.C3), norm(CTM.Cset.C4),
                norm(CTM.Tset.T1), norm(CTM.Tset.T2), norm(CTM.Tset.T3), norm(CTM.Tset.T4),
            ],
        )
    end
    return signature
end

function _corner_spectrum(C; normalize::Bool=true)
    Cmat = numind(C) == 2 ? permute(C, ((1,), (2,))) : C
    vals = abs.(collect(parent(svd_vals(Cmat))))
    sort!(vals; rev=true)
    if normalize && !isempty(vals) && vals[1] != 0
        vals ./= vals[1]
    end
    return vals
end

function graded_ctm_spectrum_signature(CTM_cell::AbstractMatrix; normalize::Bool=true)
    signature = Float64[]
    for cx in axes(CTM_cell, 1), cy in axes(CTM_cell, 2)
        CTM = CTM_cell[cx, cy]
        append!(signature, _corner_spectrum(CTM.Cset.C1; normalize))
        append!(signature, _corner_spectrum(CTM.Cset.C2; normalize))
        append!(signature, _corner_spectrum(CTM.Cset.C3; normalize))
        append!(signature, _corner_spectrum(CTM.Cset.C4; normalize))
    end
    return signature
end

function _padded_relative_error(current::AbstractVector, previous::AbstractVector)
    n = max(length(current), length(previous))
    n == 0 && return 0.0
    curr = zeros(Float64, n)
    prev = zeros(Float64, n)
    curr[1:length(current)] .= current
    prev[1:length(previous)] .= previous
    return norm(curr - prev) / max(norm(prev), eps(Float64))
end

function _ctm_signature(CTM_cell::AbstractMatrix, conv_check::Symbol)
    if conv_check === :spectrum
        return graded_ctm_spectrum_signature(CTM_cell)
    elseif conv_check === :norm
        return graded_ctm_norm_signature(CTM_cell)
    else
        error("unknown conv_check: $conv_check; expected :spectrum or :norm")
    end
end

function graded_ctm_cell_iterate(CTM_cell::AbstractMatrix, AA_cell::AbstractMatrix; maxiter::Integer=1, direction_order=(3, 4, 1, 2), chi::Union{Nothing,Int}=nothing, cutoff::Real=1e-14, conv_check::Symbol=:spectrum)
    current = CTM_cell
    signatures = Vector{Vector{Float64}}(undef, maxiter + 1)
    errors = zeros(Float64, maxiter)
    signatures[1] = _ctm_signature(current, conv_check)

    for iter in 1:maxiter
        result = graded_ctm_cell_sweep(current, AA_cell; direction_order, chi, cutoff)
        current = result.CTM
        signatures[iter + 1] = _ctm_signature(current, conv_check)
        prev = signatures[iter]
        curr = signatures[iter + 1]
        errors[iter] = _padded_relative_error(curr, prev)
    end

    return (CTM=current, signatures=signatures, errors=errors)
end
