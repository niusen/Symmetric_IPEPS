function ES_CTMRG_prepare_explicit(CTM, U_L, U_R, N, group_index; T_scale=1)
    Tleft =  CTM.Tset.T4
    Tleft=Tleft/norm(Tleft);
    Tright = CTM.Tset.T2
    Tright=Tright/norm(Tright);

    Tleft = T_scale * Tleft 
    Tright = T_scale * Tright


    @tensor O1[:] := Tleft[-3, 1, -1] * U_L[1, -2, -4]
    @tensor O2[:] := Tright[-1, 1, -3] * U_R[-4, -2, 1]

    @tensor OO[:] := O1[-2, -3, -5, 1] * O2[-1, 1, -4, -6]
    U_fuse_chichi = unitary(fuse(space(OO, 1) * space(OO, 2)), space(OO, 1) * space(OO, 2))
    @tensor OO[:] := U_fuse_chichi[-1, 1, 2] * OO[1, 2, -2, 3, 4, -4] *
        U_fuse_chichi'[3, 4, -3]

    U_fuse_DD = unitary(fuse(space(O1, 2) * space(O1, 2)), space(O1, 2)' * space(O1, 2)')
    if group_index
        @tensor O1_O1[:] := O1[-1, 1, 2, 4] * O1[2, 3, -3, 5] *
            U_fuse_DD'[1, 3, -2] * U_fuse_DD[-4, 4, 5]
        @tensor O2_O2[:] := O2[-1, 1, 2, 4] * O2[2, 3, -3, 5] *
            U_fuse_DD'[1, 3, -2] * U_fuse_DD[-4, 4, 5]
        # Keep the absolute scale from Tleft/Tright. Normalizing these grouped
        # tensors would remove the requested T_scale from the transfer matrix.

        if N == 8
            U_fuse_DD_D = unitary(
                fuse(space(O1_O1, 2) * space(O1, 2)),
                space(O1_O1, 2)' * space(O1, 2)',
            )
            @tensor O1_O1_O1[:] := O1_O1[-1, 1, 2, 4] * O1[2, 3, -3, 5] *
                U_fuse_DD_D'[1, 3, -2] * U_fuse_DD_D[-4, 4, 5]
            @tensor O2_O2_O2[:] := O2_O2[-1, 1, 2, 4] * O2[2, 3, -3, 5] *
                U_fuse_DD_D'[1, 3, -2] * U_fuse_DD_D[-4, 4, 5]
            # Keep the absolute scale from Tleft/Tright here as well.
            @tensor a_bcd_To_abc_d[:] := U_fuse_DD_D[-1, 3, 4] *
                U_fuse_DD[3, -3, 2] * U_fuse_DD'[2, 4, 1] *
                U_fuse_DD_D'[1, -2, -4]
        else
            U_fuse_DD_D = nothing
            O1_O1_O1 = nothing
            O2_O2_O2 = nothing
            a_bcd_To_abc_d = nothing
        end
    else
        O1_O1 = nothing
        O2_O2 = nothing
        U_fuse_DD_D = nothing
        O1_O1_O1 = nothing
        O2_O2_O2 = nothing
        a_bcd_To_abc_d = nothing
    end

    return (
        O1, O2, OO, U_fuse_DD, O1_O1, O2_O2, U_fuse_DD_D,
        O1_O1_O1, O2_O2_O2, a_bcd_To_abc_d,
    )
end

function CTM_T_group_action_explicit(
    O1,
    O2,
    U_fuse_DD,
    O1_O1,
    O2_O2,
    U_fuse_DD_D,
    O1_O1_O1,
    O2_O2_O2,
    a_bcd_To_abc_d,
    v0,
    N,
    kn,
    vison,
)
    if N == 4
        if vison
            op = vison_op(space(O1_O1, 3))
            @tensor v_new[:] := O1_O1[5, 1, 2, -1] * O1_O1[2, 3, 4, -2] *
                op[5, 4] * v0[1, 3, -3]
            op = vison_op(space(O2_O2, 3))
            @tensor v_new[:] := O2_O2[5, 1, 2, -1] * O2_O2[2, 3, 4, -2] *
                op[5, 4] * v_new[1, 3, -3]
        else
            @tensor v_new[:] := O1_O1[4, 1, 2, -1] * O1_O1[2, 3, 4, -2] *
                v0[1, 3, -3]
            @tensor v_new[:] := O2_O2[4, 1, 2, -1] * O2_O2[2, 3, 4, -2] *
                v_new[1, 3, -3]
        end

        @tensor v_new[:] := v_new[1, 2, -5] * U_fuse_DD'[-1, -2, 1] *
            U_fuse_DD'[-3, -4, 2]
        if kn != []
            v_new = k_projection(v_new, vison, N, kn, U_fuse_DD, U_fuse_DD_D, a_bcd_To_abc_d)
        end
        @tensor v_new[:] := v_new[1, 2, 3, 4, -3] * U_fuse_DD[-1, 1, 2] *
            U_fuse_DD[-2, 3, 4]
    elseif N == 5
        if vison
            op = vison_op(space(O1, 3))
            @tensor v_new[:] := O1_O1[7, 1, 2, -1] * O1_O1[2, 3, 4, -2] *
                O1[4, 5, 6, -3] * op[7, 6] * v0[1, 3, 5, -4]
            op = vison_op(space(O2, 3))
            @tensor v_new[:] := O2_O2[7, 1, 2, -1] * O2_O2[2, 3, 4, -2] *
                O2[4, 5, 6, -3] * op[7, 6] * v_new[1, 3, 5, -4]
        else
            @tensor v_new[:] := O1_O1[6, 1, 2, -1] * O1_O1[2, 3, 4, -2] *
                O1[4, 5, 6, -3] * v0[1, 3, 5, -4]
            @tensor v_new[:] := O2_O2[6, 1, 2, -1] * O2_O2[2, 3, 4, -2] *
                O2[4, 5, 6, -3] * v_new[1, 3, 5, -4]
        end

        @tensor v_new[:] := v_new[1, 2, -5, -6] * U_fuse_DD'[-1, -2, 1] *
            U_fuse_DD'[-3, -4, 2]
        if kn != []
            v_new = k_projection(v_new, vison, N, kn, U_fuse_DD, U_fuse_DD_D, a_bcd_To_abc_d)
        end
        @tensor v_new[:] := v_new[1, 2, 3, 4, -3, -4] * U_fuse_DD[-1, 1, 2] *
            U_fuse_DD[-2, 3, 4]
    elseif N == 6
        if vison
            op = vison_op(space(O1_O1, 3))
            @tensor v_new[:] := O1_O1[7, 1, 2, -1] * O1_O1[2, 3, 4, -2] *
                O1_O1[4, 5, 6, -3] * op[7, 6] * v0[1, 3, 5, -4]
            op = vison_op(space(O2_O2, 3))
            @tensor v_new[:] := O2_O2[7, 1, 2, -1] * O2_O2[2, 3, 4, -2] *
                O2_O2[4, 5, 6, -3] * op[7, 6] * v_new[1, 3, 5, -4]
        else
            @tensor v_new[:] := O1_O1[6, 1, 2, -1] * O1_O1[2, 3, 4, -2] *
                O1_O1[4, 5, 6, -3] * v0[1, 3, 5, -4]
            @tensor v_new[:] := O2_O2[6, 1, 2, -1] * O2_O2[2, 3, 4, -2] *
                O2_O2[4, 5, 6, -3] * v_new[1, 3, 5, -4]
        end

        @tensor v_new[:] := v_new[1, 2, 3, -7] * U_fuse_DD'[-1, -2, 1] *
            U_fuse_DD'[-3, -4, 2] * U_fuse_DD'[-5, -6, 3]
        if kn != []
            v_new = k_projection(v_new, vison, N, kn, U_fuse_DD, U_fuse_DD_D, a_bcd_To_abc_d)
        end
        @tensor v_new[:] := v_new[1, 2, 3, 4, 5, 6, -4] *
            U_fuse_DD[-1, 1, 2] * U_fuse_DD[-2, 3, 4] * U_fuse_DD[-3, 5, 6]
    elseif N == 8
        if vison
            op = vison_op(space(O1_O1, 3))
            @tensor v_new[:] := O1_O1_O1[7, 1, 2, -1] *
                O1_O1_O1[2, 3, 4, -2] * O1_O1[4, 5, 6, -3] *
                op[7, 6] * v0[1, 3, 5, -4]
            op = vison_op(space(O2_O2, 3))
            @tensor v_new[:] := O2_O2_O2[7, 1, 2, -1] *
                O2_O2_O2[2, 3, 4, -2] * O2_O2[4, 5, 6, -3] *
                op[7, 6] * v_new[1, 3, 5, -4]
        else
            @tensor v_new[:] := O1_O1_O1[6, 1, 2, -1] *
                O1_O1_O1[2, 3, 4, -2] * O1_O1[4, 5, 6, -3] *
                v0[1, 3, 5, -4]
            @tensor v_new[:] := O2_O2_O2[6, 1, 2, -1] *
                O2_O2_O2[2, 3, 4, -2] * O2_O2[4, 5, 6, -3] *
                v_new[1, 3, 5, -4]
        end

        if kn != []
            v_new = k_projection(v_new, vison, N, kn, U_fuse_DD, U_fuse_DD_D, a_bcd_To_abc_d)
        end
    end
    return v_new
end

function ES_CTMRG_ED_explicit(CTM, U_L, U_R, D, chi, N, EH_n, group_index, vison; T_scale=1)
    println("D=" * string(D))
    println("chi=" * string(chi))
    println("N=" * string(N))
    flush(stdout)

    (
        O1, O2, OO, U_fuse_DD, O1_O1, O2_O2, U_fuse_DD_D,
        O1_O1_O1, O2_O2_O2, a_bcd_To_abc_d,
    ) = ES_CTMRG_prepare_explicit(CTM, U_L, U_R, N, group_index; T_scale=T_scale)

    println("calculate ES for N=" * string(N))
    Sectors = [0, 1 / 2, 1, 3 / 2, 2, 5 / 2]
    eu_set = Vector(undef, length(Sectors))
    ks_set = Vector(undef, length(Sectors))

    for sps in eachindex(Sectors)
        if N == 4
            v_init = TensorMap(
                randn,
                space(OO, 2)' * space(OO, 2)' * space(OO, 2)' * space(OO, 2)',
                SU2Space(Sectors[sps] => 1),
            )
            v_init = permute(v_init, (1, 2, 3, 4, 5,), ())
            if group_index
                @tensor v_init[:] := v_init[1, 2, 3, 4, -3] *
                    U_fuse_DD[-1, 1, 2] * U_fuse_DD[-2, 3, 4]
            end
        elseif N == 5
            v_init = TensorMap(
                randn,
                space(OO, 2)' * space(OO, 2)' * space(OO, 2)' * space(OO, 2)' * space(OO, 2)',
                SU2Space(Sectors[sps] => 1),
            )
            v_init = permute(v_init, (1, 2, 3, 4, 5, 6,), ())
            if group_index
                @tensor v_init[:] := v_init[1, 2, 3, 4, -3, -4] *
                    U_fuse_DD[-1, 1, 2] * U_fuse_DD[-2, 3, 4]
            end
        elseif N == 6
            v_init = TensorMap(
                randn,
                space(OO, 2)' * space(OO, 2)' * space(OO, 2)' *
                    space(OO, 2)' * space(OO, 2)' * space(OO, 2)',
                SU2Space(Sectors[sps] => 1),
            )
            v_init = permute(v_init, (1, 2, 3, 4, 5, 6, 7,), ())
            if group_index
                @tensor v_init[:] := v_init[1, 2, 3, 4, 5, 6, -4] *
                    U_fuse_DD[-1, 1, 2] * U_fuse_DD[-2, 3, 4] *
                    U_fuse_DD[-3, 5, 6]
            end
        elseif N == 8
            @assert group_index == true
            v_init = TensorMap(
                randn,
                fuse(space(OO, 2)' * space(OO, 2)' * space(OO, 2)') *
                    fuse(space(OO, 2)' * space(OO, 2)' * space(OO, 2)') *
                    fuse(space(OO, 2)' * space(OO, 2)'),
                SU2Space(Sectors[sps] => 1),
            )
            v_init = permute(v_init, (1, 2, 3, 4,), ())
        end

        if norm(v_init) < 1e-12
            eu_set[sps] = []
            ks_set[sps] = []
            continue
        end

        if group_index
            contraction_group_fun(x) = CTM_T_group_action_explicit(
                O1, O2, U_fuse_DD, O1_O1, O2_O2, U_fuse_DD_D,
                O1_O1_O1, O2_O2_O2, a_bcd_To_abc_d, x, N, [], vison,
            )
            @time eu, ev = eigsolve(contraction_group_fun, v_init, EH_n, :LM,
                Arnoldi(krylovdim=EH_n * 2 + 5))
        else
            contraction_fun(x) = CTM_T_action(OO, x, N, vison)
            @time eu, ev = eigsolve(contraction_fun, v_init, EH_n, :LM,
                Arnoldi(krylovdim=EH_n * 2 + 5))
        end

        eu_set[sps] = eu
        ks_set[sps] = calculate_k(ev, N, vison, group_index, U_fuse_DD, a_bcd_To_abc_d)
        println("spin: " * string(Sectors[sps]))
        println(eu)
        flush(stdout)
    end

    ES_filenm = vison ? "ES_vison_D$(D)_chi$(chi)_N$(N).mat" :
        "ES_D$(D)_chi$(chi)_N$(N).mat"
    matwrite(ES_filenm, Dict(
        "eu_set" => eu_set,
        "Sectors" => Sectors,
        "ks_set" => ks_set,
    ); compress=false)
end

function ES_CTMRG_ED_Kprojector_explicit(CTM, U_L, U_R, D, chi, N, EH_n, group_index, vison; T_scale=1)
    println("D=" * string(D))
    println("chi=" * string(chi))
    println("N=" * string(N))
    flush(stdout)

    (
        O1, O2, OO, U_fuse_DD, O1_O1, O2_O2, U_fuse_DD_D,
        O1_O1_O1, O2_O2_O2, a_bcd_To_abc_d,
    ) = ES_CTMRG_prepare_explicit(CTM, U_L, U_R, N, group_index; T_scale=T_scale)

    println("calculate ES for N=" * string(N))
    Sectors = [0, 1 / 2, 1, 3 / 2, 2, 5 / 2]
    Ks = collect(0:N-1)
    eu_set = Matrix(undef, length(Ks), length(Sectors))

    for kk in eachindex(Ks)
        for sps in eachindex(Sectors)
            if N == 4
                v_init = TensorMap(
                    randn,
                    space(OO, 2)' * space(OO, 2)' * space(OO, 2)' * space(OO, 2)',
                    SU2Space(Sectors[sps] => 1),
                )
                v_init = permute(v_init, (1, 2, 3, 4, 5,), ())
                v_init = k_projection(v_init, vison, N, Ks[kk], U_fuse_DD, U_fuse_DD_D, a_bcd_To_abc_d)
                if group_index
                    @tensor v_init[:] := v_init[1, 2, 3, 4, -3] *
                        U_fuse_DD[-1, 1, 2] * U_fuse_DD[-2, 3, 4]
                end
            elseif N == 5
                v_init = TensorMap(
                    randn,
                    space(OO, 2)' * space(OO, 2)' * space(OO, 2)' * space(OO, 2)' * space(OO, 2)',
                    SU2Space(Sectors[sps] => 1),
                )
                v_init = permute(v_init, (1, 2, 3, 4, 5, 6,), ())
                v_init = k_projection(v_init, vison, N, Ks[kk], U_fuse_DD, U_fuse_DD_D, a_bcd_To_abc_d)
                if group_index
                    @tensor v_init[:] := v_init[1, 2, 3, 4, -3, -4] *
                        U_fuse_DD[-1, 1, 2] * U_fuse_DD[-2, 3, 4]
                end
            elseif N == 6
                v_init = TensorMap(
                    randn,
                    space(OO, 2)' * space(OO, 2)' * space(OO, 2)' *
                        space(OO, 2)' * space(OO, 2)' * space(OO, 2)',
                    SU2Space(Sectors[sps] => 1),
                )
                v_init = permute(v_init, (1, 2, 3, 4, 5, 6, 7,), ())
                v_init = k_projection(v_init, vison, N, Ks[kk], U_fuse_DD, U_fuse_DD_D, a_bcd_To_abc_d)
                if group_index
                    @tensor v_init[:] := v_init[1, 2, 3, 4, 5, 6, -4] *
                        U_fuse_DD[-1, 1, 2] * U_fuse_DD[-2, 3, 4] *
                        U_fuse_DD[-3, 5, 6]
                end
            elseif N == 8
                @assert group_index == true
                v_init = TensorMap(
                    randn,
                    fuse(space(OO, 2)' * space(OO, 2)' * space(OO, 2)') *
                        fuse(space(OO, 2)' * space(OO, 2)' * space(OO, 2)') *
                        fuse(space(OO, 2)' * space(OO, 2)'),
                    SU2Space(Sectors[sps] => 1),
                )
                v_init = permute(v_init, (1, 2, 3, 4,), ())
                v_init = k_projection(v_init, vison, N, Ks[kk], U_fuse_DD, U_fuse_DD_D, a_bcd_To_abc_d)
            end

            if norm(v_init) < 1e-12
                eu_set[kk, sps] = []
                continue
            end

            if group_index
                contraction_group_fun(x) = CTM_T_group_action_explicit(
                    O1, O2, U_fuse_DD, O1_O1, O2_O2, U_fuse_DD_D,
                    O1_O1_O1, O2_O2_O2, a_bcd_To_abc_d, x, N, Ks[kk], vison,
                )
                @time eu, ev = eigsolve(contraction_group_fun, v_init, EH_n, :LM,
                    Arnoldi(krylovdim=EH_n * 2 + 5))
            else
                contraction_fun(x) = CTM_T_action(OO, x, N, vison)
                @time eu, ev = eigsolve(contraction_fun, v_init, EH_n, :LM,
                    Arnoldi(krylovdim=EH_n * 2 + 5))
            end
            eu_set[kk, sps] = eu

            println("momentum: " * string(Ks[kk]))
            println("spin: " * string(Sectors[sps]))
            println(eu)
            flush(stdout)
        end
    end

    ES_filenm = vison ? "ES_Kprojector_vison_D$(D)_chi$(chi)_N$(N).mat" :
        "ES_Kprojector_D$(D)_chi$(chi)_N$(N).mat"
    matwrite(ES_filenm, Dict(
        "eu_set" => eu_set,
        "Sectors" => Sectors,
        "Ks" => Ks,
    ); compress=false)
end
