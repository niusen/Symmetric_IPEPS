"""
CTMRG-differentiated fixed-space variational optimization for square-J1 iPEPS.

This optimizes the site tensors without changing their SU(2) virtual spaces.
The optimization uses the repository's Square_iPEPS_immutable/OptimKit LBFGS
path and its existing tangent conversion and manifold operations.
"""

function square_J1_variational_energy_cell(
    A_set::AbstractMatrix,
    H::TensorMap,
    environment_chi::Int,
    ctm_setting;
    J1::Real=1,
    return_ctm_info::Bool=false,
)
    cell_Lx, cell_Ly = size(A_set)
    A_normalized = map(A_set) do entry
        A = entry isa TensorMap ? entry : entry.T
        _square_fu_normalize(A)
    end
    A_cell = square_fu_cell_to_tuple(A_normalized)
    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    result = CTMRG_cell(A_cell, environment_chi, init, [], ctm_setting)
    CTM = result[1]

    total = 0.0
    for cx in 1:cell_Lx, cy in 1:cell_Ly, direction in (:x, :y)
        site1 = CartesianIndex(cx, cy)
        site2 = direction === :x ?
            CartesianIndex(mod1(cx + 1, cell_Lx), cy) :
            CartesianIndex(cx, mod1(cy + 1, cell_Ly))
        bond = SquareJ1CellBond(direction, site1, site2)
        A1, A2 = A_normalized[site1], A_normalized[site2]
        rho = _square_fu_two_site_density_cell(
            CTM, A1, A1, A2, A2, bond, cell_Lx, cell_Ly,
        )
        rho_norm = real(@tensor rho[1, 2, 1, 2])
        rho_norm > 0 || throw(ArgumentError(
            "non-positive two-site norm on $direction bond at ($cx, $cy)",
        ))
        total += real(@tensor rho[1, 2, 3, 4] * H[1, 2, 3, 4]) / rho_norm
    end
    energy = J1 * total / (cell_Lx * cell_Ly)
    if return_ctm_info
        return (
            energy=energy,
            ctm_iterations=length(result) == 8 ? result[7] : missing,
            ctm_error=length(result) == 8 ? result[8] : missing,
        )
    end
    return energy
end

"""Convert a square iPEPS cell to the immutable ansatz used for AD."""
square_J1_immutable_cell(A_set::AbstractMatrix) =
    [Square_iPEPS_immutable(A_set[cx, cy]) for cx in axes(A_set, 1), cy in axes(A_set, 2)]

"""Return plain, normalized tensors for the Full Update state format."""
square_J1_tensor_cell(x::AbstractMatrix{Square_iPEPS_immutable}) =
    [_square_fu_normalize(x[cx, cy].T) for cx in axes(x, 1), cy in axes(x, 2)]

# These are the Square_iPEPS_immutable arithmetic rules from
# square_large_cell_optimization.jl, needed by the same OptimKit operations.
Base.:*(a::Square_iPEPS_immutable, beta::Number) = Square_iPEPS_immutable(a.T * beta)
Base.:*(beta::Number, a::Square_iPEPS_immutable) = Square_iPEPS_immutable(a.T * beta)
Base.:+(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable) =
    Square_iPEPS_immutable(a.T + b.T)
Base.:-(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable) =
    Square_iPEPS_immutable(a.T - b.T)

"""
    square_J1_variational_optimize_cell(A_initial, H, chi, grad_ctm, ls_ctm; ...)

Use the repository's OptimKit LBFGS(8) path, with independent CTMRG settings
for AD and energy verification. Save only a new best, converged LS energy.
"""
function square_J1_variational_optimize_cell(
    A_initial::AbstractMatrix,
    H::TensorMap,
    environment_chi::Int,
    grad_ctm_setting,
    ls_ctm_setting;
    J1::Real=1,
    max_iterations::Int=10,
    gradient_tolerance::Real=1.0e-6,
    callback=(A_set, evaluation, energy, ctm_iterations, ctm_error) -> nothing,
)
    _square_fu_validate_cell(A_initial)
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    x = square_J1_immutable_cell(map(_square_fu_normalize, A_initial))
    best_energy = Ref(Inf)
    evaluation_count = Ref(0)

    function verify_and_save(x_trial, evaluation)
        measured = square_J1_variational_energy_cell(
            x_trial, H, environment_chi, ls_ctm_setting;
            J1, return_ctm_info=true,
        )
        E = measured.energy
        converged = !ismissing(measured.ctm_error) &&
            measured.ctm_error <= ls_ctm_setting.CTM_conv_tol
        println(
            "  LS energy=$E, ctm_ite_num=$(measured.ctm_iterations), " *
            "ctm_ite_err=$(measured.ctm_error), ctm_converged=$converged",
        )
        flush(stdout)
        if converged && isfinite(E) && E < best_energy[]
            best_energy[] = E
            callback(
                square_J1_tensor_cell(x_trial), evaluation, E,
                measured.ctm_iterations, measured.ctm_error,
            )
        end
        return nothing
    end

    verify_and_save(x, 0)
    isfinite(best_energy[]) || error(
        "initial LS CTMRG did not converge; increase ctm_max_iterations or loosen ctm_tolerance",
    )

    function costfun_grad(x_trial::Matrix{Square_iPEPS_immutable})
        evaluation_count[] += 1
        println("\nOptimKit fg evaluation $(evaluation_count[])")
        flush(stdout)
        out = Zygote.withgradient(
            y -> square_J1_variational_energy_cell(
                y, H, environment_chi, grad_ctm_setting; J1,
            ),
            x_trial,
        )
        E = out.val
        gradient = NamedTuple_to_Struc_cell_optimkit(out.grad[1], x_trial)
        grad_norm = sqrt(max(my_inner(x_trial, gradient, gradient), 0.0))
        println("  trial energy E=$E, trial grad_norm=$grad_norm")
        flush(stdout)
        if isfinite(E) && E < best_energy[]
            verify_and_save(x_trial, evaluation_count[])
        end
        return E, gradient
    end

    x_opt, fx, gx, numfg, grad_history = optimize(
        costfun_grad,
        x,
        LBFGS(8; maxiter=max_iterations, gradtol=gradient_tolerance, verbosity=3);
        inner=my_inner,
        retract=my_retract,
        scale! = my_scale!,
        add! = my_add!,
    )
    verify_and_save(x_opt, evaluation_count[])
    return square_J1_tensor_cell(x_opt), fx, best_energy[], numfg, grad_history
end
