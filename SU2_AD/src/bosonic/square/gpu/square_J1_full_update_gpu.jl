"""
GPU orchestration for the bosonic square-lattice J1 Full Update.

The persistent iPEPS state and checkpoints stay on CPU. CTMRG, the local
two-site Full Update, and observables are moved independently to their chosen
devices, following the triangular Hofstadter-Hubbard Full Update workflow.
"""

# The local square-lattice FU uses only a 2x1 or 1x2 cluster.  Its reduced
# contraction needs the CTM tensors and the four per-site fusion maps, but not
# the full-cell double-layer tensors `environment.AA`.  Avoid copying those
# unused tensors to the FU device, which is especially important at D=12.
_square_J1_fu_environment_view(environment) = (
    CTM=environment.CTM,
    U_L=environment.U_L,
    U_D=environment.U_D,
    U_R=environment.U_R,
    U_U=environment.U_U,
)

_square_J1_energy_environment_view(environment) = (CTM=environment.CTM,)

function square_J1_gpu_environment_cell(
    A_set::AbstractMatrix,
    environment_chi::Int,
    ctm_setting;
    device::AbstractString="cuda:0",
    ctm_memory::SquareJ1CTMMemorySettings=SquareJ1CTMMemorySettings(),
)
    A_run = square_J1_to_device(device, A_set)
    cell_Lx, cell_Ly = _square_fu_validate_cell(A_run)
    global Lx = cell_Lx
    global Ly = cell_Ly
    global chi = environment_chi
    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    result = square_J1_CTMRG_cell_offload(
        square_fu_cell_to_tuple(A_run), environment_chi, init, [], ctm_setting;
        memory=ctm_memory,
    )
    CTM, AA, U_L, U_D, U_R, U_U = result[1:6]
    ite_num, ite_err = length(result) == 8 ? (result[7], result[8]) : (missing, missing)
    environment_run = (; CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err,
                        Lx=cell_Lx, Ly=cell_Ly)
    environment = square_J1_to_cpu(environment_run)
    A_run = nothing
    environment_run = nothing
    result = CTM = AA = U_L = U_D = U_R = U_U = nothing
    square_J1_reclaim_device_memory!(aggressive=true)
    return environment
end

function square_J1_gpu_update_bond(
    A_set::AbstractMatrix,
    environment,
    gate::TensorMap,
    bond::SquareJ1CellBond;
    device::AbstractString="cuda:0",
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
)
    A_run = square_J1_to_device(device, A_set)
    environment_run = square_J1_to_device(
        device, _square_J1_fu_environment_view(environment),
    )
    gate_run = square_J1_to_device(device, gate)
    A1_run, A2_run, report = square_J1_full_update_cell_bond_als(
        A_run, environment_run, gate_run, bond; settings,
    )
    A1 = square_J1_to_cpu(A1_run)
    A2 = square_J1_to_cpu(A2_run)
    A_run = nothing
    environment_run = nothing
    gate_run = nothing
    A1_run = nothing
    A2_run = nothing
    square_J1_reclaim_device_memory!(aggressive=true)
    return A1, A2, report
end

function square_J1_gpu_energy_cell(
    A_set::AbstractMatrix,
    environment;
    J1::Real=1,
    device::AbstractString="cuda:0",
    low_memory::Bool=true,
    debug_memory::Bool=false,
)
    A_run = square_J1_to_device(device, A_set)
    environment_run = square_J1_to_device(
        device, _square_J1_energy_environment_view(environment),
    )
    # Clear cached workspaces left by the preceding Full Update/CTMRG before
    # the first observable contraction.  Live A/CTM tensors remain valid.
    square_J1_reclaim_device_memory!(aggressive=true)
    debug_memory && square_J1_print_device_memory(
        "Observable baseline after A/CTM transfer:",
    )
    energies = square_J1_energy_cell_device(
        A_run, environment_run; J1, low_memory, debug_memory,
    )
    energies = square_J1_to_cpu(energies)
    A_run = nothing
    environment_run = nothing
    square_J1_reclaim_device_memory!(aggressive=true)
    return energies
end

"""
GPU-oriented streaming two-site density matrix.

Only one open double layer is live at a time.  It is immediately absorbed into
the corresponding 2x1/1x2 half environment and both the tensor and CUDA pool
workspace are released before the second site is constructed.
"""
function _square_J1_two_site_density_cell_device_low_memory(
    CTM,
    A_1::TensorMap,
    A_2::TensorMap,
    bond::SquareJ1CellBond,
    cell_Lx::Int,
    cell_Ly::Int;
    debug_memory::Bool=false,
)
    x1, y1 = Tuple(bond.site1)
    anchor_x, anchor_y = x1 - 1, y1 - 1
    bond_label = "$(bond.direction) bond ($(x1),$(y1))"

    debug_memory && square_J1_print_device_memory(
        "  $bond_label before first AA_open:",
    )
    AA_1, U_physical_1 = build_square_cross_double_layer_open(A_1, A_1)
    debug_memory && square_J1_print_device_memory(
        "  $bond_label after first AA_open:",
    )
    if bond.direction === :x
        half_1 = _square_fu_x_left_half_cell(
            anchor_x, anchor_y, CTM, AA_1, cell_Lx, cell_Ly,
        )
    elseif bond.direction === :y
        half_1 = _square_fu_y_upper_half_cell(
            anchor_x, anchor_y, CTM, AA_1, cell_Lx, cell_Ly,
        )
    else
        throw(ArgumentError("bond direction must be :x or :y"))
    end
    AA_1 = nothing
    square_J1_reclaim_device_memory!()
    debug_memory && square_J1_print_device_memory(
        "  $bond_label after first half/reclaim:",
    )

    AA_2, U_physical_2 = build_square_cross_double_layer_open(A_2, A_2)
    debug_memory && square_J1_print_device_memory(
        "  $bond_label after second AA_open:",
    )
    if bond.direction === :x
        half_2 = _square_fu_x_right_half_cell(
            anchor_x, anchor_y, CTM, AA_2, cell_Lx, cell_Ly,
        )
    else
        half_2 = _square_fu_y_lower_half_cell(
            anchor_x, anchor_y, CTM, AA_2, cell_Lx, cell_Ly,
        )
    end
    AA_2 = nothing
    square_J1_reclaim_device_memory!()
    debug_memory && square_J1_print_device_memory(
        "  $bond_label after second half/reclaim:",
    )

    @tensor rho_fused[:] := half_1[1, 2, 3, -1] * half_2[1, 2, 3, -2]
    half_1 = nothing
    half_2 = nothing
    square_J1_reclaim_device_memory!()
    @tensor rho[:] := rho_fused[1, 2] *
        U_physical_1[-1, -3, 1] *
        U_physical_2[-2, -4, 2]
    rho_fused = nothing
    U_physical_1 = nothing
    U_physical_2 = nothing
    square_J1_reclaim_device_memory!()
    debug_memory && square_J1_print_device_memory(
        "  $bond_label after rho construction/reclaim:",
    )
    return rho
end

function square_J1_energy_cell_device(
    A_set::AbstractMatrix,
    environment;
    J1::Real=1,
    low_memory::Bool=true,
    debug_memory::Bool=false,
)
    cell_Lx, cell_Ly = _square_fu_validate_cell(A_set)
    H_Heisenberg, _, _, _, _ = Hamiltonians(space(A_set[1, 1], 1))
    H = permute(H_Heisenberg, (1, 2), (3, 4))
    H = square_J1_to_storage_like(H, A_set[1, 1])
    Ex = zeros(Float64, cell_Lx, cell_Ly)
    Ey = zeros(Float64, cell_Lx, cell_Ly)
    for cx in 1:cell_Lx, cy in 1:cell_Ly
        for direction in (:x, :y)
            site1 = CartesianIndex(cx, cy)
            site2 = direction === :x ?
                CartesianIndex(mod1(cx + 1, cell_Lx), cy) :
                CartesianIndex(cx, mod1(cy + 1, cell_Ly))
            bond = SquareJ1CellBond(direction, site1, site2)
            A1, A2 = A_set[site1], A_set[site2]
            rho = if low_memory
                _square_J1_two_site_density_cell_device_low_memory(
                    environment.CTM, A1, A2, bond, cell_Lx, cell_Ly;
                    debug_memory,
                )
            else
                _square_fu_two_site_density_cell(
                    environment.CTM, A1, A1, A2, A2, bond, cell_Lx, cell_Ly,
                )
            end
            norm_rho = real(@tensor rho[1, 2, 1, 2])
            norm_rho != 0 || throw(ArgumentError(
                "zero norm for bond $direction at ($cx,$cy)",
            ))
            energy_numerator = @tensor rho[1, 2, 3, 4] * H[1, 2, 3, 4]
            energy = J1 * real(energy_numerator) / norm_rho
            direction === :x ? (Ex[cx, cy] = energy) : (Ey[cx, cy] = energy)
            if low_memory
                rho = nothing
                square_J1_reclaim_device_memory!()
                debug_memory && square_J1_print_device_memory(
                    "  $direction bond ($cx,$cy) after energy/reclaim:",
                )
            end
        end
    end
    H = nothing
    return (
        energy_per_site=(sum(Ex) + sum(Ey)) / (cell_Lx * cell_Ly),
        Ex=Ex,
        Ey=Ey,
    )
end

function square_J1_full_update_cell_gpu_sweep(
    A_set::AbstractMatrix,
    environment_chi::Int,
    gate::TensorMap,
    ctm_setting;
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
    bond_groups=nothing,
    initial_environment=nothing,
    ctm_device::AbstractString="cuda:0",
    full_update_device::AbstractString="cuda:0",
    ctm_memory::SquareJ1CTMMemorySettings=SquareJ1CTMMemorySettings(),
)
    cell_Lx, cell_Ly = _square_fu_validate_cell(A_set)
    settings.refresh_environment || throw(ArgumentError(
        "square cell Full Update requires refresh_environment=true",
    ))
    groups = isnothing(bond_groups) ? square_J1_bond_groups(cell_Lx, cell_Ly) : bond_groups
    A_current = copy(A_set)
    environment = isnothing(initial_environment) ?
        square_J1_gpu_environment_cell(
            A_current, environment_chi, ctm_setting; device=ctm_device, ctm_memory,
        ) : initial_environment
    if settings.verbose && isnothing(initial_environment)
        println(
            "ctm_ite_num= $(environment.ite_num), " *
            "ctm_ite_err= $(environment.ite_err)",
        )
        flush(stdout)
    end
    reports = NamedTuple[]

    for (group_index, group) in pairs(groups), bond in group
        A1, A2, report = square_J1_gpu_update_bond(
            A_current, environment, gate, bond;
            device=full_update_device,
            settings,
        )
        A_current[bond.site1] = A1
        A_current[bond.site2] = A2
        push!(reports, merge(report, (group=group_index,)))

        # Reconstruct CTMRG after every bond, matching the CPU and triangular
        # Full Update algorithms. No old CTM is reused.
        environment = square_J1_gpu_environment_cell(
            A_current, environment_chi, ctm_setting; device=ctm_device, ctm_memory,
        )
        if settings.verbose
            println(
                "ctm_ite_num= $(environment.ite_num), " *
                "ctm_ite_err= $(environment.ite_err)",
            )
            flush(stdout)
        end
    end
    return A_current, environment, reports
end

function square_J1_full_update_cell_gpu(
    A_set::AbstractMatrix,
    environment_chi::Int,
    tau::Real,
    dt::Real,
    ctm_setting;
    J1::Real=1,
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
    ctm_device::AbstractString="cuda:0",
    full_update_device::AbstractString="cuda:0",
    ctm_memory::SquareJ1CTMMemorySettings=SquareJ1CTMMemorySettings(),
    callback=nothing,
)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    tau >= 0 || throw(ArgumentError("tau must be non-negative"))
    steps_float = tau / dt
    nsteps = round(Int, steps_float)
    isapprox(steps_float, nsteps; atol=1.0e-12, rtol=1.0e-12) ||
        throw(ArgumentError("tau/dt must be an integer, got $steps_float"))

    cell_Lx, cell_Ly = _square_fu_validate_cell(A_set)
    gate = prepare_gate_Heisenberg(J1 * dt, space(A_set[1, 1], 1))
    gate = square_J1_to_scalartype_like(gate, A_set[1, 1])
    groups = square_J1_bond_groups(cell_Lx, cell_Ly)
    A_current = copy(A_set)
    environment = square_J1_gpu_environment_cell(
        A_current, environment_chi, ctm_setting; device=ctm_device, ctm_memory,
    )
    if settings.verbose
        println(
            "ctm_ite_num= $(environment.ite_num), " *
            "ctm_ite_err= $(environment.ite_err)",
        )
        println("Periodic bond-group sizes: $(map(length, groups))")
        flush(stdout)
    end

    history = Vector{Vector{NamedTuple}}()
    for step in 1:nsteps
        settings.verbose && println("iteration $step")
        A_current, environment, reports = square_J1_full_update_cell_gpu_sweep(
            A_current,
            environment_chi,
            gate,
            ctm_setting;
            settings,
            bond_groups=groups,
            initial_environment=environment,
            ctm_device,
            full_update_device,
            ctm_memory,
        )
        push!(history, reports)
        isnothing(callback) || callback(A_current, environment, step, reports)
    end
    return A_current, environment, history
end
