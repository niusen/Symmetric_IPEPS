"""
Settings for the one-tensor square-lattice J1 full update.

The local optimization keeps the converged CTM environment fixed.  Therefore
the reverse-mode tape contains only a two-site contraction, not the CTMRG
iterations themselves.
"""
Base.@kwdef struct SquareJ1FullUpdateSettings
    maxiter::Int = 20
    gradient_tolerance::Float64 = 1.0e-8
    loss_tolerance::Float64 = 1.0e-12
    initial_step::Float64 = 0.2
    backtracking_factor::Float64 = 0.5
    armijo::Float64 = 1.0e-4
    minimum_step::Float64 = 1.0e-10
    metric_floor::Float64 = 0.0
    refresh_environment::Bool = true
    verbose::Bool = true
end

function _square_fu_check_tensor_pair(A_bra::TensorMap, A_ket::TensorMap)
    numind(A_bra) == 5 || throw(ArgumentError("square iPEPS tensors must have five legs (L,D,R,U,physical)"))
    numind(A_ket) == 5 || throw(ArgumentError("square iPEPS tensors must have five legs (L,D,R,U,physical)"))
    for leg in 1:5
        space(A_bra, leg) == space(A_ket, leg) ||
            throw(SpaceMismatch("bra and ket spaces differ on leg $leg"))
    end
    return nothing
end

"""
    build_square_cross_double_layer_open(A_bra, A_ket)

Construct a bosonic double-layer tensor whose physical bra and ket indices are
left open and fused into the fifth leg.  For `A_bra === A_ket` this is the
cross-state generalization of `build_double_layer_open` in `square_model.jl`.

Both tensors use the repository convention `(L,D,R,U,physical)`.
"""
function build_square_cross_double_layer_open(A_bra::TensorMap, A_ket::TensorMap)
    _square_fu_check_tensor_pair(A_bra, A_ket)

    A_space = permute(A_ket, (1, 2), (3, 4, 5))
    U_L = @ignore_derivatives unitary(
        fuse(space(A_space, 1)' ⊗ space(A_space, 1)),
        space(A_space, 1)' ⊗ space(A_space, 1),
    ) * (1 + 0im)
    U_D = @ignore_derivatives unitary(
        fuse(space(A_space, 2)' ⊗ space(A_space, 2)),
        space(A_space, 2)' ⊗ space(A_space, 2),
    ) * (1 + 0im)
    U_R = @ignore_derivatives unitary(
        space(A_space, 3) ⊗ space(A_space, 3)',
        fuse(space(A_space, 3)' ⊗ space(A_space, 3)),
    ) * (1 + 0im)
    U_U = @ignore_derivatives unitary(
        space(A_space, 4) ⊗ space(A_space, 4)',
        fuse(space(A_space, 4)' ⊗ space(A_space, 4)),
    ) * (1 + 0im)

    # Split bra and ket at the same virtual cut.  The unitaries depend only on
    # spaces, so they are deliberately kept outside the AD tape.
    A_bra_adj = permute(A_bra', (1, 2, 5), (3, 4))
    U_bra = @ignore_derivatives unitary(
        fuse(space(A_bra_adj, 1) ⊗ space(A_bra_adj, 2) ⊗ space(A_bra_adj, 3)),
        space(A_bra_adj, 1) ⊗ space(A_bra_adj, 2) ⊗ space(A_bra_adj, 3),
    ) * (1 + 0im)
    v_bra = U_bra * A_bra_adj
    u_bra = U_bra'

    U_ket = @ignore_derivatives unitary(
        fuse(space(A_ket, 1) ⊗ space(A_ket, 2)),
        space(A_ket, 1) ⊗ space(A_ket, 2),
    ) * (1 + 0im)
    v_ket = U_ket * permute(A_ket, (1, 2), (3, 4, 5))
    u_ket = U_ket'

    u_bra = permute(u_bra, (1, 2, 3, 4), ())
    u_ket = permute(u_ket, (1, 2, 3), ())
    V_bra = space(v_bra, 1)
    V_ket = space(v_ket, 1)
    U_mid = @ignore_derivatives unitary(
        fuse(V_bra ⊗ V_ket),
        V_bra ⊗ V_ket,
    ) * (1 + 0im)

    @tensor double_LD[:] := u_bra[-1, -2, -3, 1] * U_mid'[1, -4, -5]
    @tensor double_LD[:] := double_LD[-1, -3, -5, 1, -6] * u_ket[-2, -4, 1]

    v_bra = permute(v_bra, (1, 2, 3), ())
    v_ket = permute(v_ket, (1, 2, 3, 4))
    @tensor double_RU[:] := U_mid[-1, -2, 1] * v_ket[1, -3, -4, -5]
    @tensor double_RU[:] := v_bra[1, -2, -4] * double_RU[-1, 1, -3, -5, -6]

    double_LD = permute(double_LD, (1, 2), (3, 4, 5, 6))
    double_LD = U_L * double_LD
    double_LD = permute(double_LD, (2, 3), (1, 4, 5))
    double_LD = U_D * double_LD
    double_LD = permute(double_LD, (2, 1, 3, 4), ())

    double_RU = permute(double_RU, (1, 2, 3, 6), (4, 5))
    double_RU = double_RU * U_U
    @tensor double_RU[:] := double_RU[-1, 1, 2, -4, -3] * U_R[1, 2, -2]

    V_physical_bra = space(A_bra, 5)
    V_physical_ket = space(A_ket, 5)
    V_physical_pair = @ignore_derivatives fuse(V_physical_bra' ⊗ V_physical_ket)
    U_physical = @ignore_derivatives unitary(
        V_physical_pair,
        V_physical_bra' ⊗ V_physical_ket,
    ) * (1 + 0im)

    @tensor AA_open[:] := double_LD[-1, -2, 1, 3] *
                          double_RU[3, -3, -4, 2] *
                          U_physical[-5, 1, 2]

    return AA_open, U_physical'
end

function _square_fu_two_site_density(
    CTM,
    A_bra_1::TensorMap,
    A_ket_1::TensorMap,
    A_bra_2::TensorMap,
    A_ket_2::TensorMap,
    direction::Symbol,
)
    AA_1, U_physical_1 = build_square_cross_double_layer_open(A_bra_1, A_ket_1)
    AA_2, U_physical_2 = build_square_cross_double_layer_open(A_bra_2, A_ket_2)

    if direction === :x
        rho_fused = ob_2sites_x(CTM, AA_1, AA_2)
    elseif direction === :y
        rho_fused = ob_2sites_y(CTM, AA_1, AA_2)
    else
        throw(ArgumentError("direction must be :x or :y, got $direction"))
    end

    # Output order: bra-site-1, bra-site-2, ket-site-1, ket-site-2.
    @tensor rho[:] := rho_fused[1, 2] *
                      U_physical_1[-1, -3, 1] *
                      U_physical_2[-2, -4, 2]
    return rho
end

function _square_fu_target_norm(A_old::TensorMap, CTM, gate::TensorMap, direction::Symbol)
    rho_old = _square_fu_two_site_density(CTM, A_old, A_old, A_old, A_old, direction)
    gate_norm = gate' * gate
    value = @tensor rho_old[1, 2, 3, 4] * gate_norm[1, 2, 3, 4]
    return real(value)
end

"""
    square_J1_full_update_fidelity(A_new, A_old, CTM, gate, direction;
                                   target_norm=nothing, metric_floor=0)

Normalized fidelity between the candidate two-site state and the state obtained
by applying `gate` to the old two-site state.  The surrounding infinite tensor
network is represented by the fixed `CTM` environment.
"""
function square_J1_full_update_fidelity(
    A_new::TensorMap,
    A_old::TensorMap,
    CTM,
    gate::TensorMap,
    direction::Symbol;
    target_norm=nothing,
    metric_floor::Real=0.0,
)
    rho_new = _square_fu_two_site_density(CTM, A_new, A_new, A_new, A_new, direction)
    norm_new = real(@tensor rho_new[1, 2, 1, 2])

    rho_overlap = _square_fu_two_site_density(CTM, A_new, A_old, A_new, A_old, direction)
    overlap = @tensor rho_overlap[1, 2, 3, 4] * gate[1, 2, 3, 4]

    norm_target = isnothing(target_norm) ?
        _square_fu_target_norm(A_old, CTM, gate, direction) : target_norm
    denominator = abs(norm_new * norm_target)
    denominator > metric_floor ||
        throw(ArgumentError("the CTM fidelity denominator is non-positive or too small: $denominator"))
    return real(abs2(overlap) / denominator)
end

function square_J1_full_update_loss(
    A_new::TensorMap,
    A_old::TensorMap,
    CTM,
    gate::TensorMap,
    direction::Symbol;
    target_norm=nothing,
    metric_floor::Real=0.0,
)
    fidelity = square_J1_full_update_fidelity(
        A_new,
        A_old,
        CTM,
        gate,
        direction;
        target_norm=target_norm,
        metric_floor=metric_floor,
    )
    return 1 - fidelity
end

function _square_fu_normalize(A::TensorMap)
    nA = norm(A)
    nA > 0 || throw(ArgumentError("cannot normalize a zero iPEPS tensor"))
    return A / nA
end

"""
    square_J1_full_update_bond(A_old, CTM, gate, direction; settings)

Perform one environment-aware variational truncation for a horizontal or
vertical J1 bond.  A backtracking steepest-descent solver is used because the
one-tensor SU(2) manifold is linear and TensorKit automatically restricts the
gradient to the allowed symmetry blocks.
"""
function square_J1_full_update_bond(
    A_old::TensorMap,
    CTM,
    gate::TensorMap,
    direction::Symbol;
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
)
    direction in (:x, :y) || throw(ArgumentError("direction must be :x or :y"))
    target_norm = _square_fu_target_norm(A_old, CTM, gate, direction)
    target_norm > 0 ||
        throw(ArgumentError("the CTM target norm is non-positive: $target_norm"))

    objective(A) = square_J1_full_update_loss(
        A,
        A_old,
        CTM,
        gate,
        direction;
        target_norm=target_norm,
        metric_floor=settings.metric_floor,
    )

    A_current = _square_fu_normalize(A_old)
    loss_current = objective(A_current)
    loss_initial = loss_current
    loss_history = Float64[loss_current]
    gradient_norm = Inf
    accepted_steps = 0

    settings.verbose && println("FU $direction: initial loss = $loss_current")
    for iteration in 1:settings.maxiter
        gradient_tensor = Zygote.gradient(objective, A_current)[1]
        gradient_tensor === nothing && error("Zygote returned no gradient for the full-update objective")
        gradient_norm = norm(gradient_tensor)
        settings.verbose && println(
            "FU $direction iteration $iteration: loss=$loss_current, |grad|=$gradient_norm",
        )
        gradient_norm <= settings.gradient_tolerance && break

        direction_tensor = -gradient_tensor
        slope = real(dot(gradient_tensor, direction_tensor))
        step = settings.initial_step
        accepted = false
        while step >= settings.minimum_step
            A_trial = _square_fu_normalize(A_current + step * direction_tensor)
            loss_trial = objective(A_trial)
            if isfinite(loss_trial) && loss_trial <= loss_current + settings.armijo * step * slope
                improvement = loss_current - loss_trial
                A_current = A_trial
                loss_current = loss_trial
                push!(loss_history, loss_current)
                accepted_steps += 1
                accepted = true
                improvement <= settings.loss_tolerance && return A_current, (
                    direction=direction,
                    loss_initial=loss_initial,
                    loss_final=loss_current,
                    fidelity=1 - loss_current,
                    iterations=iteration,
                    accepted_steps=accepted_steps,
                    gradient_norm=gradient_norm,
                    target_norm=target_norm,
                    loss_history=loss_history,
                )
                break
            end
            step *= settings.backtracking_factor
        end
        accepted || break
    end

    report = (
        direction=direction,
        loss_initial=loss_initial,
        loss_final=loss_current,
        fidelity=1 - loss_current,
        iterations=length(loss_history) - 1,
        accepted_steps=accepted_steps,
        gradient_norm=gradient_norm,
        target_norm=target_norm,
        loss_history=loss_history,
    )
    return A_current, report
end

function _square_fu_environment(A::TensorMap, chi::Int, ctm_setting; initial_CTM=nothing)
    if isnothing(initial_CTM)
        init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
        CTM0 = []
    else
        init = initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true)
        CTM0 = initial_CTM
    end
    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err = CTMRG(A, chi, init, CTM0, ctm_setting)
    return (
        CTM=CTM,
        AA=AA,
        U_L=U_L,
        U_D=U_D,
        U_R=U_R,
        U_U=U_U,
        ite_num=ite_num,
        ite_err=ite_err,
    )
end

"""
    square_J1_full_update_sweep(A, chi, gate, ctm_setting; ...)

Update all requested bond directions and refresh the CTM environment after each
accepted bond update.  The default `(:x, :y)` is one first-order Trotter sweep.
"""
function square_J1_full_update_sweep(
    A::TensorMap,
    chi::Int,
    gate::TensorMap,
    ctm_setting;
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
    directions=(:x, :y),
    initial_environment=nothing,
)
    environment = isnothing(initial_environment) ?
        _square_fu_environment(A, chi, ctm_setting) : initial_environment
    reports = NamedTuple[]

    for direction in directions
        A, report = square_J1_full_update_bond(A, environment.CTM, gate, direction; settings=settings)
        push!(reports, report)
        if settings.refresh_environment
            environment = _square_fu_environment(
                A,
                chi,
                ctm_setting;
                initial_CTM=environment.CTM,
            )
        end
    end

    if !settings.refresh_environment
        environment = _square_fu_environment(A, chi, ctm_setting; initial_CTM=environment.CTM)
    end
    return A, environment, reports
end

"""
    square_J1_full_update(A, chi, tau, dt, ctm_setting; J1=1, ...)

Run imaginary-time Full Update for the one-tensor bosonic square-lattice J1
Heisenberg iPEPS.  `callback(A, environment, step, reports)` is called after
each horizontal-plus-vertical sweep and can be used for measurements/checkpoint
saving.
"""
function square_J1_full_update(
    A::TensorMap,
    chi::Int,
    tau::Real,
    dt::Real,
    ctm_setting;
    J1::Real=1,
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
    callback=nothing,
)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    tau >= 0 || throw(ArgumentError("tau must be non-negative"))
    steps_float = tau / dt
    nsteps = round(Int, steps_float)
    isapprox(steps_float, nsteps; atol=1.0e-12, rtol=1.0e-12) ||
        throw(ArgumentError("tau/dt must be an integer, got $steps_float"))

    gate = prepare_gate_Heisenberg(J1 * dt, space(A, 1))
    environment = nothing
    history = Vector{Vector{NamedTuple}}()
    for step in 1:nsteps
        settings.verbose && println("square J1 FU sweep $step/$nsteps, dt=$dt")
        A, environment, reports = square_J1_full_update_sweep(
            A,
            chi,
            gate,
            ctm_setting;
            settings=settings,
            initial_environment=environment,
        )
        push!(history, reports)
        isnothing(callback) || callback(A, environment, step, reports)
    end
    return A, environment, history
end
