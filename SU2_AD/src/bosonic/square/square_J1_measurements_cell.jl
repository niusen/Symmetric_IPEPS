"""
Reusable Simple-Update and CTMRG energy measurements for a bosonic
square-lattice J1 iPEPS cell.

Include the square spin operator, Settings/CTMRG files, Simple Update files,
and `full_update_J1_cell.jl` before this file.
"""

function square_J1_simple_x_energy(
    T_set,
    lambda_x,
    lambda_y,
    cx::Int,
    cy::Int,
    H,
)
    cell_Lx, cell_Ly = size(T_set)
    site1 = (cx, cy)
    site2 = (mod1(cx + 1, cell_Lx), cy)
    lambda1 = lambda_x[site1...]
    lambda2 = lambda_y[site1...]
    lambda3 = lambda_y[site1[1], mod1(site1[2] + 1, cell_Ly)]
    lambda4 = lambda_y[site2...]
    lambda5 = lambda_x[mod1(site2[1] + 1, cell_Lx), site2[2]]
    lambda6 = lambda_y[site2[1], mod1(site2[2] + 1, cell_Ly)]
    T1, T2 = T_set[site1...], T_set[site2...]
    @tensor T1_env[:] := T1[1, 2, -3, 3, -5] *
        lambda1[-1, 1] * lambda2[2, -2] * lambda3[-4, 3]
    @tensor T2_env[:] := T2[-1, 1, 2, 3, -5] *
        lambda4[1, -2] * lambda5[2, -3] * lambda6[-4, 3]
    _, singular1, right1 = tsvd(permute(T1_env, (1, 2, 4), (3, 5)))
    left2, singular2, _ = tsvd(permute(T2_env, (1, 5), (2, 3, 4)))
    keep1 = singular1 * right1
    keep2 = left2 * singular2
    @tensor psi[:] := keep1[-1, 1, -3] * keep2[1, -2, -4]
    # The physical indices are restored to the order expected by H.
    psi = permute(psi, (1, 4), (3, 2))
    physical_isometry = unitary(domain(psi), codomain(H))
    H_psi = physical_isometry * H * physical_isometry'
    return real(dot(psi, psi * H_psi) / dot(psi, psi))
end

function square_J1_simple_y_energy(
    T_set,
    lambda_x,
    lambda_y,
    cx::Int,
    cy::Int,
    H,
)
    cell_Lx, cell_Ly = size(T_set)
    upper = (cx, mod1(cy + 1, cell_Ly))
    lower = (cx, cy)
    lambda1 = lambda_x[upper...]
    lambda2 = lambda_x[mod1(upper[1] + 1, cell_Lx), upper[2]]
    lambda3 = lambda_y[upper[1], mod1(upper[2] + 1, cell_Ly)]
    lambda4 = lambda_x[lower...]
    lambda5 = lambda_y[lower...]
    lambda6 = lambda_x[mod1(lower[1] + 1, cell_Lx), lower[2]]
    T1, T2 = T_set[upper...], T_set[lower...]
    @tensor T1_env[:] := T1[1, -2, 2, 3, -5] *
        lambda1[-1, 1] * lambda2[2, -3] * lambda3[-4, 3]
    @tensor T2_env[:] := T2[1, 2, 3, -4, -5] *
        lambda4[-1, 1] * lambda5[2, -2] * lambda6[3, -3]
    _, singular1, right1 = tsvd(permute(T1_env, (1, 3, 4), (2, 5)))
    left2, singular2, _ = tsvd(permute(T2_env, (4, 5), (1, 2, 3)))
    keep1 = singular1 * right1
    keep2 = left2 * singular2
    @tensor psi[:] := keep1[-1, 1, -3] * keep2[1, -2, -4]
    psi = permute(psi, (1, 4), (3, 2))
    physical_isometry = unitary(domain(psi), codomain(H))
    H_psi = physical_isometry * H * physical_isometry'
    return real(dot(psi, psi * H_psi) / dot(psi, psi))
end

"""
    square_J1_simple_energy_cell(T_set, lambda_x, lambda_y; J1=1)

Estimate every nearest-neighbour bond energy using the Simple-Update lambda
environment.  This is inexpensive and useful while evolving, but is not a
replacement for the final CTMRG energy.
"""
function square_J1_simple_energy_cell(
    T_set,
    lambda_x,
    lambda_y;
    J1::Real=1,
)
    size(T_set) == size(lambda_x) == size(lambda_y) || throw(
        DimensionMismatch("T_set, lambda_x, and lambda_y must have the same size"),
    )
    cell_Lx, cell_Ly = size(T_set)
    H_Heisenberg, _, _, _, _ = Hamiltonians(space(T_set[1, 1], 1))
    H = permute(H_Heisenberg, (1, 2), (3, 4))
    Ex = [
        J1 * square_J1_simple_x_energy(
            T_set, lambda_x, lambda_y, cx, cy, H,
        ) for cx in 1:cell_Lx, cy in 1:cell_Ly
    ]
    Ey = [
        J1 * square_J1_simple_y_energy(
            T_set, lambda_x, lambda_y, cx, cy, H,
        ) for cx in 1:cell_Lx, cy in 1:cell_Ly
    ]
    return (
        energy_per_site=(sum(Ex) + sum(Ey)) / (cell_Lx * cell_Ly),
        Ex=Ex,
        Ey=Ey,
    )
end

function square_J1_default_ctm_settings(;
    tolerance::Real=1.0e-6,
    maxiter::Int=120,
    verbose::Bool=false,
)
    settings = LS_CTMRG_settings()
    settings.CTM_conv_tol = tolerance
    settings.CTM_ite_nums = maxiter
    settings.CTM_trun_tol = 1.0e-8
    settings.svd_lanczos_tol = 1.0e-8
    settings.projector_strategy = "4x4"
    settings.conv_check = "singular_value"
    settings.CTM_ite_info = verbose
    settings.CTM_conv_info = true
    settings.CTM_trun_svd = false
    settings.construct_double_layer = true
    settings.grad_checkpoint = false
    return settings
end

function square_J1_prepare_ctm_globals!(
    A_set,
    environment_chi::Int,
    ctm_settings;
    multiplet_tolerance::Real=1.0e-5,
    cell_method::AbstractString="continuous_update",
)
    global Lx = size(A_set, 1)
    global Ly = size(A_set, 2)
    global chi = environment_chi
    global multiplet_tol = multiplet_tolerance
    global projector_trun_tol = ctm_settings.CTM_trun_tol
    global backward_settings = Backward_settings()
    global algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
    algrithm_CTMRG_settings.CTM_cell_ite_method = cell_method
    return nothing
end

function square_J1_environment_cell(
    A_set,
    environment_chi::Int,
    ctm_settings;
    multiplet_tolerance::Real=1.0e-5,
    cell_method::AbstractString="continuous_update",
    initial_CTM=nothing,
)
    square_J1_prepare_ctm_globals!(
        A_set,
        environment_chi,
        ctm_settings;
        multiplet_tolerance,
        cell_method,
    )
    return _square_fu_environment_cell(
        A_set,
        environment_chi,
        ctm_settings;
        initial_CTM,
    )
end

"""
    square_J1_measure_ctm_energy_step(A_set, chi; initial_CTM=nothing, ...)

Run one CTMRG energy-measurement step and return `(measurement, CTM)`.  A CTM
from a smaller previous `chi` may be supplied as the initial boundary for an
increasing-chi scan.  Only the converged C/T boundary is retained; CTMRG
double layers and fusion tensors are released before the low-memory 2×1/1×2
energy contractions.
"""
function square_J1_measure_ctm_energy_step(
    A_set,
    environment_chi::Int;
    J1::Real=1,
    tolerance::Real=1.0e-6,
    maxiter::Int=120,
    verbose::Bool=false,
    multiplet_tolerance::Real=1.0e-5,
    cell_method::AbstractString="continuous_update",
    initial_CTM=nothing,
)
    ctm_settings = square_J1_default_ctm_settings(;
        tolerance,
        maxiter,
        verbose,
    )
    environment = square_J1_environment_cell(
        A_set,
        environment_chi,
        ctm_settings;
        multiplet_tolerance,
        cell_method,
        initial_CTM,
    )

    CTM = environment.CTM
    ctm_iterations = environment.ite_num
    ctm_error = environment.ite_err
    environment = nothing
    GC.gc()

    energy = square_J1_energy_cell(
        A_set,
        (CTM=CTM,);
        J1,
        low_memory=true,
    )
    measurement = merge(energy, (
        chi=environment_chi,
        ctm_iterations,
        ctm_error,
        reused_initial_CTM=!isnothing(initial_CTM),
    ))
    return measurement, CTM
end

"""
    square_J1_measure_ctm_energy_cell(A_set, chi; ...)

Reconstruct CTMRG from scratch and measure all x/y J1 bonds.  `A_set` is used
directly; Simple-Update lambda tensors must not be absorbed into it again.
"""
function square_J1_measure_ctm_energy_cell(
    A_set,
    environment_chi::Int;
    J1::Real=1,
    tolerance::Real=1.0e-6,
    maxiter::Int=120,
    verbose::Bool=false,
    multiplet_tolerance::Real=1.0e-5,
    cell_method::AbstractString="continuous_update",
)
    measurement, CTM = square_J1_measure_ctm_energy_step(
        A_set,
        environment_chi;
        J1,
        tolerance,
        maxiter,
        verbose,
        multiplet_tolerance,
        cell_method,
    )
    CTM = nothing
    GC.gc()
    return measurement
end
