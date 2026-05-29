const CHIRAL_PAIR_LARGE_CELL_VIRTUAL_ISOMETRY_CACHE = Dict{Any, Any}()

function chiral_pair_large_cell_virtual_isometry(Vsmall, Vbig)
    @assert dim(Vbig) >= dim(Vsmall) "Cannot embed a larger virtual space into a smaller one."
    key = (Vsmall, Vbig)
    if !haskey(CHIRAL_PAIR_LARGE_CELL_VIRTUAL_ISOMETRY_CACHE, key)
        W = isometry(Vbig, Vsmall)
        @assert norm(W' * W - unitary(Vsmall, Vsmall)) < 1e-12 "Virtual embedding isometry does not satisfy W' * W = I."
        CHIRAL_PAIR_LARGE_CELL_VIRTUAL_ISOMETRY_CACHE[key] = W
    end
    return CHIRAL_PAIR_LARGE_CELL_VIRTUAL_ISOMETRY_CACHE[key]
end

function embed_chiral_pair_large_cell_tensor_virtual_space(A, Vbig)
    Vsmall = space(A, 1)
    if Vsmall == Vbig
        return permute(A, (1, 2, 3, 4, 5,))
    end
    @assert space(A, 2) == Vsmall "Only tensors with equal first two virtual spaces are supported."
    @assert space(A, 3) == Vsmall' "Expected third virtual leg to be the dual of the first leg."
    @assert space(A, 4) == Vsmall' "Expected fourth virtual leg to be the dual of the first leg."

    W = chiral_pair_large_cell_virtual_isometry(Vsmall, Vbig)
    @tensor A_big[:] := W[-1, 1] * W[-2, 2] * W'[3, -3] * W'[4, -4] *
        A[1, 2, 3, 4, -5]
    return permute(A_big, (1, 2, 3, 4, 5,))
end

function _chiral_pair_large_cell_tensor_with_noise(A, init_noise::Real, init_complex_tensor::Bool)
    if init_noise == 0
        return permute(A, (1, 2, 3, 4, 5,))
    end
    if init_complex_tensor
        A_noise = TensorMap(randn, codomain(A), domain(A)) +
            im * TensorMap(randn, codomain(A), domain(A))
    else
        A_noise = TensorMap(randn, codomain(A), domain(A))
    end
    A_new = A + A_noise * init_noise * norm(A) / norm(A_noise)
    return permute(A_new, (1, 2, 3, 4, 5,))
end

function _as_chiral_pair_large_cell_tensor(x)
    if isa(x, Square_iPEPS) || isa(x, Square_iPEPS_immutable)
        return x.T
    end
    return x
end

function _tuple_cell_size(A_cell)
    return (length(A_cell), length(A_cell[1]))
end

function _loaded_chiral_pair_large_cell_tensor(data, cx::Int, cy::Int)
    if haskey(data, "x")
        x = data["x"]
        if size(x) == (Lx, Ly)
            return _as_chiral_pair_large_cell_tensor(x[cx, cy])
        elseif size(x) == (1, 1)
            return _as_chiral_pair_large_cell_tensor(x[1, 1])
        end
        error("Loaded x has size " * string(size(x)) * ", incompatible with target cell " * string((Lx, Ly)))
    elseif haskey(data, "A_cell")
        A_cell = data["A_cell"]
        if _tuple_cell_size(A_cell) == (Lx, Ly)
            return _as_chiral_pair_large_cell_tensor(A_cell[cx][cy])
        elseif _tuple_cell_size(A_cell) == (1, 1)
            return _as_chiral_pair_large_cell_tensor(A_cell[1][1])
        end
        error("Loaded A_cell has size " * string(_tuple_cell_size(A_cell)) *
            ", incompatible with target cell " * string((Lx, Ly)))
    elseif haskey(data, "A")
        A = data["A"]
        if isa(A, AbstractArray)
            if size(A) == (Lx, Ly)
                return _as_chiral_pair_large_cell_tensor(A[cx, cy])
            elseif size(A) == (1, 1)
                return _as_chiral_pair_large_cell_tensor(A[1, 1])
            end
        end
        return _as_chiral_pair_large_cell_tensor(A)
    end
    error("Initial state file must contain `x`, `A_cell`, or `A`.")
end

function initial_SU2_chiral_pair_large_cell_state(
    Vspace,
    init_statenm::String="nothing",
    init_noise::Real=0,
    init_complex_tensor::Bool=true,
)
    Vp_pair = chiral_pair_physical_space()
    state = Matrix{Square_iPEPS}(undef, Lx, Ly)

    if init_statenm == "nothing"
        println("Random initial large-cell chiral-pair SU2 state")
        for cx in 1:Lx, cy in 1:Ly
            if init_complex_tensor
                A = TensorMap(randn, Vspace * Vspace * Vspace' * Vspace', Vp_pair) +
                    im * TensorMap(randn, Vspace * Vspace * Vspace' * Vspace', Vp_pair)
            else
                A = TensorMap(randn, Vspace * Vspace * Vspace' * Vspace', Vp_pair)
            end
            state[cx, cy] = Square_iPEPS(permute(A, (1, 2, 3, 4, 5,)))
        end
        return state
    end

    println("load large-cell chiral-pair SU2 state: " * init_statenm)
    data = load(init_statenm)
    for cx in 1:Lx, cy in 1:Ly
        A = _loaded_chiral_pair_large_cell_tensor(data, cx, cy)
        @assert space(A, 5) == Vp_pair' "Loaded physical space is not the chiral-pair physical leg Rep[SU2](0=>1, 1=>1)'."
        if space(A, 1) != Vspace
            println("Embed loaded chiral-pair virtual space at site " * string((cx, cy)) *
                " from " * string(space(A, 1)) * " to " * string(Vspace))
            A = embed_chiral_pair_large_cell_tensor_virtual_space(A, Vspace)
        end
        @assert space(A, 1) == Vspace "Loaded virtual space does not match requested Vspace after embedding."
        A = _chiral_pair_large_cell_tensor_with_noise(A, init_noise, init_complex_tensor)
        state[cx, cy] = Square_iPEPS(A)
    end
    return state
end

function chiral_pair_large_cell_to_A_cell(x)
    A_cell = initial_tuple_cell(Lx, Ly)
    for cx in 1:Lx, cy in 1:Ly
        A = _as_chiral_pair_large_cell_tensor(x[cx, cy])
        A = A / norm(A)
        A_cell = fill_tuple(A_cell, A, cx, cy)
    end
    return A_cell
end

function chiral_pair_large_cell_ctmrg(A_cell, chi_value, init, init_CTM, ctm_setting)
    out = CTMRG_cell(A_cell, chi_value, init, init_CTM, ctm_setting)
    if length(out) == 8
        return out
    elseif length(out) == 6
        CTM_cell, AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell = out
        return CTM_cell, AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell, missing, missing
    end
    error("Unexpected CTMRG_cell return length: " * string(length(out)))
end

function _chiral_pair_large_cell_ctm_converged(ite_err, ctm_setting)
    if ismissing(ite_err)
        return true
    end
    return ite_err <= ctm_setting.CTM_conv_tol
end

function evaluate_chiral_pair_large_cell_terms(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting)
    J1 = parameters["J1"]
    J2 = parameters["J2"]
    Jchi = parameters["Jchi"]

    AA_open_cell = initial_tuple_cell(Lx, Ly)
    for cx in 1:Lx, cy in 1:Ly
        if ctm_setting.grad_checkpoint
            AA_open, _ = Zygote.checkpointed(build_double_layer_open, A_cell[cx][cy])
        else
            AA_open, _ = build_double_layer_open(A_cell[cx][cy])
        end
        AA_open_cell = fill_tuple(AA_open_cell, AA_open, cx, cy)
    end

    V_s = @ignore_derivatives space(A_cell[1][1], 5)
    V_ss = @ignore_derivatives fuse(V_s' * V_s)
    U_s_s = @ignore_derivatives unitary(V_ss, V_s' * V_s)'
    @ignore_derivatives chiral_pair_fuse_unitary()

    e12_A_set = zeros(ComplexF64, 4, Lx, Ly)
    e31_A_set = zeros(ComplexF64, 4, Lx, Ly)
    e23_A_set = zeros(ComplexF64, 4, Lx, Ly)
    chi_A_set = zeros(ComplexF64, 4, Lx, Ly)
    e12_B_set = zeros(ComplexF64, 4, Lx, Ly)
    e31_B_set = zeros(ComplexF64, 4, Lx, Ly)
    e23_B_set = zeros(ComplexF64, 4, Lx, Ly)
    chi_B_set = zeros(ComplexF64, 4, Lx, Ly)
    E_triangle_set = zeros(ComplexF64, 4, Lx, Ly)

    E_total = 0
    for cx in 1:Lx, cy in 1:Ly
        pos_LU = [mod1(cx + 1, Lx), mod1(cy + 1, Ly)]
        pos_RU = [mod1(cx + 2, Lx), mod1(cy + 1, Ly)]
        pos_LD = [mod1(cx + 1, Lx), mod1(cy + 2, Ly)]
        pos_RD = [mod1(cx + 2, Lx), mod1(cy + 2, Ly)]

        rho1 = ob_LU_RU_LD_cell(cx, cy, CTM_cell, AA_cell[pos_RD[1]][pos_RD[2]],
            AA_open_cell[pos_LU[1]][pos_LU[2]], AA_open_cell[pos_RU[1]][pos_RU[2]],
            AA_open_cell[pos_LD[1]][pos_LD[2]])

        rho2 = ob_LD_RU_RD_cell(cx, cy, CTM_cell, AA_cell[pos_LU[1]][pos_LU[2]],
            AA_open_cell[pos_LD[1]][pos_LD[2]], AA_open_cell[pos_RU[1]][pos_RU[2]],
            AA_open_cell[pos_RD[1]][pos_RD[2]])
        rho2 = permute(rho2, (3, 1, 2,))

        rho3 = ob_LU_LD_RD_cell(cx, cy, CTM_cell, AA_cell[pos_RU[1]][pos_RU[2]],
            AA_open_cell[pos_LU[1]][pos_LU[2]], AA_open_cell[pos_LD[1]][pos_LD[2]],
            AA_open_cell[pos_RD[1]][pos_RD[2]])
        rho3 = permute(rho3, (2, 1, 3,))

        rho4 = ob_LU_RU_RD_cell(cx, cy, CTM_cell, AA_cell[pos_LD[1]][pos_LD[2]],
            AA_open_cell[pos_LU[1]][pos_LU[2]], AA_open_cell[pos_RU[1]][pos_RU[2]],
            AA_open_cell[pos_RD[1]][pos_RD[2]])
        rho4 = permute(rho4, (2, 3, 1,))

        for (tid, rho) in enumerate((rho1, rho2, rho3, rho4))
            t = _chiral_pair_triangle_expectations(rho, U_s_s)
            etri = _chiral_pair_weight_triangle_terms(t, parameters)

            @ignore_derivatives e12_A_set[tid, cx, cy] = t.e12_A
            @ignore_derivatives e31_A_set[tid, cx, cy] = t.e31_A
            @ignore_derivatives e23_A_set[tid, cx, cy] = t.e23_A
            @ignore_derivatives chi_A_set[tid, cx, cy] = t.chi_A
            @ignore_derivatives e12_B_set[tid, cx, cy] = t.e12_B
            @ignore_derivatives e31_B_set[tid, cx, cy] = t.e31_B
            @ignore_derivatives e23_B_set[tid, cx, cy] = t.e23_B
            @ignore_derivatives chi_B_set[tid, cx, cy] = t.chi_B
            @ignore_derivatives E_triangle_set[tid, cx, cy] = etri
            E_total += real(etri)
        end
    end

    norm_cell = Lx * Ly
    nn_A = real((sum(e12_A_set) + sum(e31_A_set)) / (4 * norm_cell))
    nn_B = real((sum(e12_B_set) + sum(e31_B_set)) / (4 * norm_cell))
    nnn_A = real(sum(e23_A_set) / (2 * norm_cell))
    nnn_B = real(sum(e23_B_set) / (2 * norm_cell))
    chirality_A = real(sum(chi_A_set) / norm_cell)
    chirality_B = real(sum(chi_B_set) / norm_cell)

    obs = (
        e12_A=e12_A_set,
        e31_A=e31_A_set,
        e23_A=e23_A_set,
        chi_A=chi_A_set,
        e12_B=e12_B_set,
        e31_B=e31_B_set,
        e23_B=e23_B_set,
        chi_B=chi_B_set,
        E_triangle=E_triangle_set,
        nn_A=nn_A,
        nn_B=nn_B,
        nnn_A=nnn_A,
        nnn_B=nnn_B,
        chirality_A=chirality_A,
        chirality_B=chirality_B,
        weighted_nn=real(J1 * (nn_A + nn_B)),
        weighted_nnn=real(J2 * (nnn_A + nnn_B)),
        weighted_chirality=real(Jchi * (chirality_A - chirality_B)),
    )
    return real(E_total) / norm_cell, obs
end

function print_chiral_pair_large_cell_observables(obs)
    total = obs.weighted_nn + obs.weighted_nnn + obs.weighted_chirality
    println("  large-cell chiral-pair observables:")
    println("    <NN Heisenberg> A/B   = " * string((obs.nn_A, obs.nn_B)) *
        ", contribution=" * string(obs.weighted_nn))
    println("    <NNN Heisenberg> A/B  = " * string((obs.nnn_A, obs.nnn_B)) *
        ", contribution=" * string(obs.weighted_nnn))
    println("    <chirality> A/B       = " * string((obs.chirality_A, obs.chirality_B)) *
        ", Jchi*(A-B)=" * string(obs.weighted_chirality))
    println("    term contribution sum = " * string(total))
    flush(stdout)
    return total
end

function chiral_pair_large_cell_energy_basic(x, ctm_setting, init, init_CTM)
    A_cell = chiral_pair_large_cell_to_A_cell(x)
    CTM_cell, AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell, ite_num, ite_err =
        chiral_pair_large_cell_ctmrg(A_cell, chi, init, init_CTM, ctm_setting)
    E, obs = evaluate_chiral_pair_large_cell_terms(parameters, A_cell, AA_cell, CTM_cell, ctm_setting)
    return E, obs, ite_num, ite_err, CTM_cell, A_cell
end

function chiral_pair_large_cell_energy_with_observables(x, ctm_setting, init, init_CTM)
    return chiral_pair_large_cell_energy_basic(x, ctm_setting, init, init_CTM)
end

function mutable_chiral_pair_large_cell(x::Matrix{Square_iPEPS})
    return deepcopy(x)
end

function mutable_chiral_pair_large_cell(x::Matrix{Square_iPEPS_immutable})
    y = Matrix{Square_iPEPS}(undef, size(x, 1), size(x, 2))
    for cc in eachindex(x)
        y[cc] = Square_iPEPS(x[cc].T)
    end
    return y
end

function immutable_chiral_pair_large_cell(x::Matrix{Square_iPEPS})
    y = Matrix{Square_iPEPS_immutable}(undef, size(x, 1), size(x, 2))
    for cc in eachindex(x)
        y[cc] = Square_iPEPS_convert(x[cc])
    end
    return y
end

function save_chiral_pair_large_cell_best(x, E, obs, ite_num, ite_err)
    x_mut = normalize_ansatz(mutable_chiral_pair_large_cell(x))
    A_cell = chiral_pair_large_cell_to_A_cell(x_mut)
    global save_filenm
    jldsave(
        save_filenm;
        x=x_mut,
        A_cell=A_cell,
        energy=E,
        parameters=parameters,
        observables=obs,
        Lx=Lx,
        Ly=Ly,
        D=D,
        chi=chi,
        ctm_ite_num=ite_num,
        ctm_ite_err=ite_err,
        physical_space=string(space(A_cell[1][1], 5)),
    )
end

function _chiral_pair_large_cell_line_search_initial_condition()
    global CTM_tem
    if optim_setting.linesearch_CTM_method == "from_converged_CTM" && @isdefined(CTM_tem)
        return initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true), deepcopy(CTM_tem)
    elseif optim_setting.linesearch_CTM_method == "restart" ||
            optim_setting.linesearch_CTM_method == "from_converged_CTM"
        return initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true), []
    end
    error("Unknown linesearch_CTM_method=" * string(optim_setting.linesearch_CTM_method))
end

function chiral_pair_large_cell_f(x::Matrix{Square_iPEPS})
    init, CTM0 = _chiral_pair_large_cell_line_search_initial_condition()
    E, obs, ite_num, ite_err, CTM, A_cell =
        chiral_pair_large_cell_energy_with_observables(x, LS_ctm_setting, init, CTM0)

    converged = _chiral_pair_large_cell_ctm_converged(ite_err, LS_ctm_setting)
    println("E= " * string(E) *
        ", E_triangle= " * string(real.(obs.E_triangle[:])) *
        ", ctm_ite_num= " * string(ite_num) *
        ", ctm_ite_err= " * string(ite_err) *
        ", ctm_converged= " * string(converged))
    flush(stdout)

    if !converged
        println("Reject line-search trial because CTMRG did not converge.")
        flush(stdout)
        return Inf
    end

    global CTM_tem
    CTM_tem = deepcopy(CTM)

    global E_history
    if E < minimum(E_history)
        E_history = vcat(E_history, E)
        save_chiral_pair_large_cell_best(x, E, obs, ite_num, ite_err)
        println("Saved best large-cell chiral-pair state to " * save_filenm)
        print_chiral_pair_large_cell_observables(obs)

        global starting_time
        now_time = now()
        elapsed = Dates.canonicalize(
            Dates.CompoundPeriod(Dates.DateTime(now_time) - Dates.DateTime(starting_time))
        )
        println("Time consumed: " * string(elapsed))
        flush(stdout)
    end
    return E
end

function chiral_pair_large_cell_cost_fun_optimkit(x::Matrix{Square_iPEPS_immutable})
    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    E, obs, ite_num, ite_err, CTM, A_cell =
        chiral_pair_large_cell_energy_basic(x, grad_ctm_setting, init, [])

    println("  trial energy E= " * string(E) *
        ", E_triangle= " * string(real.(obs.E_triangle[:])) *
        ", ctm_ite_num= " * string(ite_num) *
        ", ctm_ite_err= " * string(ite_err))
    flush(stdout)

    global E_tem, CTM_tem
    E_tem = deepcopy(E)
    CTM_tem = deepcopy(CTM)
    return E
end

function chiral_pair_large_cell_get_grad(x0::Matrix{Square_iPEPS})
    x0 = normalize_ansatz(x0)
    x = immutable_chiral_pair_large_cell(x0)
    out = Zygote.withgradient(y -> chiral_pair_large_cell_cost_fun_optimkit(y), x)
    E = out.val
    grad_imm = NamedTuple_to_Struc_cell_optimkit(out.grad[1], x)
    grad = mutable_chiral_pair_large_cell(grad_imm)
    grad_norm = norm(grad)
    println("norm of grad = " * string(grad_norm))
    flush(stdout)
    global CTM_tem
    return E, grad, CTM_tem
end

function chiral_pair_large_cell_g!(gvec::Matrix{Square_iPEPS}, x)
    println("compute large-cell chiral-pair grad")
    global E_tem, CTM_tem
    E_tem, grad, CTM_tem = chiral_pair_large_cell_get_grad(x)
    for cc in eachindex(gvec)
        setindex!(gvec, grad[cc], cc)
    end
    return gvec
end

function chiral_pair_large_cell_fg!(gvec, x)
    chiral_pair_large_cell_g!(gvec, x)
    return chiral_pair_large_cell_f(x)
end

function gdoptimize_chiral_pair_large_cell(
    f,
    g!,
    fg!,
    x0::Matrix{Square_iPEPS},
    linesearch;
    maxiter::Int=500,
    g_rtol::Float64=1e-8,
    g_atol::Float64=1e-16,
)
    println("large-cell chiral-pair square CSL optimization")
    println("D=" * string(D))
    println("chi=" * string(chi))
    println("cell=" * string((Lx, Ly)))
    flush(stdout)

    x = deepcopy(x0)
    gvec = similar(x)
    g!(gvec, x)
    fx = f(x)
    gnorm = norm(gvec)
    gtol = max(g_rtol * gnorm, g_atol)

    phi(alpha) = f(x + alpha * s)
    function dphi(alpha)
        g!(gvec, x + alpha * s)
        return real(dot(gvec, s))
    end
    function phidphi(alpha)
        phi_value = fg!(gvec, x + alpha * s)
        dphi_value = real(dot(gvec, s))
        return (phi_value, dphi_value)
    end

    s = similar(gvec)
    iter = 0
    while iter < maxiter && gnorm > gtol
        println("optim iteration " * string(iter) *
            ", fx= " * string(fx) *
            ", grad_norm= " * string(gnorm))
        flush(stdout)

        x = normalize_ansatz(x)
        iter += 1
        s = (-1) * gvec
        dphi_0 = real(dot(s, gvec))
        alpha, fx = linesearch(phi, dphi, phidphi, 1.0, fx, dphi_0)
        println("accepted backtracking alpha = " * string(alpha))
        flush(stdout)

        x = x + alpha * s
        g!(gvec, x)
        gnorm = norm(gvec)
    end
    return fx, x, iter
end

function _scale_chiral_pair_large_cell_immutable(a::Square_iPEPS_immutable, beta::Number)
    return Square_iPEPS_immutable(a.T * beta)
end

function _add_chiral_pair_large_cell_immutable(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable)
    return Square_iPEPS_immutable(a.T + b.T)
end

function _sub_chiral_pair_large_cell_immutable(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable)
    return Square_iPEPS_immutable(a.T - b.T)
end

Base.:*(a::Square_iPEPS_immutable, beta::Number) = _scale_chiral_pair_large_cell_immutable(a, beta)
Base.:*(beta::Number, a::Square_iPEPS_immutable) = _scale_chiral_pair_large_cell_immutable(a, beta)
Base.:+(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable) = _add_chiral_pair_large_cell_immutable(a, b)
Base.:-(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable) = _sub_chiral_pair_large_cell_immutable(a, b)

function chiral_pair_large_cell_costfun_grad_optimkit(x::Matrix{Square_iPEPS_immutable})
    OPTIMKIT_FG_COUNT[] += 1
    println("\nOptimKit fg evaluation $(OPTIMKIT_FG_COUNT[])")
    println("  This is a cost/gradient evaluation, often a line-search trial; it is not necessarily an accepted optimization step.")
    flush(stdout)

    out = Zygote.withgradient(y -> chiral_pair_large_cell_cost_fun_optimkit(y), x)
    E = out.val
    grad = NamedTuple_to_Struc_cell_optimkit(out.grad[1], x)

    grad_norm = sqrt(max(my_inner(x, grad, grad), 0.0))
    println("  trial grad_norm = " * string(grad_norm))
    flush(stdout)

    global E_history
    if E < minimum(E_history)
        state = mutable_chiral_pair_large_cell(x)
        init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
        E_save, obs, ite_num, ite_err, CTM, A_cell =
            chiral_pair_large_cell_energy_with_observables(state, LS_ctm_setting, init, [])
        if _chiral_pair_large_cell_ctm_converged(ite_err, LS_ctm_setting) && E_save < minimum(E_history)
            E_history = vcat(E_history, E_save)
            save_chiral_pair_large_cell_best(state, E_save, obs, ite_num, ite_err)
            println("  accepted new best energy = " * string(E_save))
            println("  Saved best large-cell chiral-pair state to " * save_filenm *
                ", ctm_ite_num= " * string(ite_num) *
                ", ctm_ite_err= " * string(ite_err))
            print_chiral_pair_large_cell_observables(obs)

            global starting_time
            now_time = now()
            elapsed = Dates.canonicalize(
                Dates.CompoundPeriod(Dates.DateTime(now_time) - Dates.DateTime(starting_time))
            )
            println("  Time consumed: " * string(elapsed))
            flush(stdout)
        else
            println("  Do not save this trial because LS CTMRG did not converge or did not improve the best LS energy.")
            flush(stdout)
        end
    end
    return E, grad
end

function chiral_pair_large_cell_optimkit_op(x; maxiter::Int=500, verbosity::Int=3)
    x_opt, fx, gx, numfg, grad_history = optimize(
        chiral_pair_large_cell_costfun_grad_optimkit,
        x,
        LBFGS(8; maxiter=maxiter, verbosity=verbosity);
        inner=my_inner,
        retract=my_retract,
        scale! = my_scale!,
        add! = my_add!,
    )
    return x_opt, fx, gx, numfg, grad_history
end

function stochastic_optimize_chiral_pair_large_cell(
    x0::Matrix{Square_iPEPS},
    delta::Real,
    maxiter::Int,
    gtol::Real,
)
    println("stochastic large-cell chiral-pair optimization")
    println("D=" * string(D))
    println("chi=" * string(chi))
    println("cell=" * string((Lx, Ly)))
    println("delta=" * string(delta))
    flush(stdout)

    x = deepcopy(x0)
    gvec = similar(x)
    gnorm = Inf
    iter = 0
    while iter < maxiter && gnorm > gtol
        println("\nstochastic iteration " * string(iter))
        x = normalize_ansatz(x)

        gvec = chiral_pair_large_cell_g!(gvec, x)
        gnorm = norm(gvec)
        random_step = get_random_grad(gvec, delta)
        x_updated = x - random_step

        println("norm of grad = " * string(gnorm))
        println("norm of random grad step = " * string(norm(x_updated - x)))
        flush(stdout)

        E_updated = chiral_pair_large_cell_f(x_updated)
        if isfinite(E_updated)
            x = x_updated
        else
            println("Reject stochastic step because the trial energy is not finite.")
            flush(stdout)
        end
        iter += 1
    end
    return x
end
