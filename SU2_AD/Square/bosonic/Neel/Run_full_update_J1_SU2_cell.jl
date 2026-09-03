"""
Server launcher for the bosonic square-lattice J1 Full Update.

Edit only the configuration block below, then run

    julia Run_full_update_J1_SU2_cell.jl

No command-line parameters are required.  This launcher translates the Julia
configuration into the interface used by `Full_update_J1_SU2_cell.jl`.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Random named initial state.  The recommended small parity-resolved seed is
# :minimal_y_staggered, with Veven=0⊕1 and Vodd=1/2 on a 2x2 matching.
initial_state_kind = :minimal_y_staggered

# Set this to a JLD2 path to continue from a saved Simple/Full Update state.
# Relative paths are resolved from this file's directory.  Leave as `nothing`
# to construct `initial_state_kind` directly.
initial_state_file = nothing
# initial_state_file = "simple_update_virtual_space_scan/results/<case>/final.jld2"

cell_Lx = 2
cell_Ly = 2
random_seed = 666

# Dmax is the ordinary state-counting dimension, not D*.  Dmax=12 permits the
# paper's D*=4 even/odd spaces to emerge from the small initial state.
Dmax = 4
multiplet_tolerance = 1.0e-5

environment_chi = 60
imaginary_time = 5
time_step = 0.02
J1 = 1.0

ctm_tolerance = 1.0e-6
ctm_max_iterations = 50
ctm_cell_method = "continuous_update"

local_max_iterations = 10
gradient_tolerance = 1.0e-8
loss_tolerance = 1.0e-12
initial_line_search_step = 0.2
verbose = true

save_file = "FullUpdate_J1_$(cell_Lx)x$(cell_Ly)_Dmax_$(Dmax)_chi_$(environment_chi).jld2"

# Only used when initial_state_kind=:homogeneous and initial_state_file=nothing.
homogeneous_initial_D = 3

# ---------------------------------------------------------------------------
# Launch the existing Full Update driver
# ---------------------------------------------------------------------------

function set_full_update_environment!(name, value)
    ENV[name] = string(value)
    return nothing
end

set_full_update_environment!("FU_LX", cell_Lx)
set_full_update_environment!("FU_LY", cell_Ly)
set_full_update_environment!("FU_SEED", random_seed)
set_full_update_environment!("FU_DMAX", Dmax)
set_full_update_environment!("FU_MULTIPLET_TOL", multiplet_tolerance)
set_full_update_environment!("FU_CHI", environment_chi)
set_full_update_environment!("FU_TAU", imaginary_time)
set_full_update_environment!("FU_DT", time_step)
set_full_update_environment!("FU_J1", J1)
set_full_update_environment!("FU_CTM_TOL", ctm_tolerance)
set_full_update_environment!("FU_CTM_MAXITER", ctm_max_iterations)
set_full_update_environment!("FU_CTM_CELL_METHOD", ctm_cell_method)
set_full_update_environment!("FU_LOCAL_MAXITER", local_max_iterations)
set_full_update_environment!("FU_GRAD_TOL", gradient_tolerance)
set_full_update_environment!("FU_LOSS_TOL", loss_tolerance)
set_full_update_environment!("FU_INITIAL_STEP", initial_line_search_step)
set_full_update_environment!("FU_REFRESH_CTM", true)
set_full_update_environment!("FU_VERBOSE", verbose)
set_full_update_environment!("FU_SAVE", isabspath(save_file) ? save_file : joinpath(@__DIR__, save_file))

if isnothing(initial_state_file)
    set_full_update_environment!("FU_INIT", "nothing")
    set_full_update_environment!("FU_INIT_KIND", initial_state_kind)
    if initial_state_kind === :homogeneous
        set_full_update_environment!("FU_D", homogeneous_initial_D)
    else
        pop!(ENV, "FU_D", nothing)
    end
else
    state_path = isabspath(initial_state_file) ?
        initial_state_file : joinpath(@__DIR__, initial_state_file)
    isfile(state_path) || error("initial_state_file does not exist: $state_path")
    pop!(ENV, "FU_D", nothing)
    set_full_update_environment!("FU_INIT", state_path)
    # The loaded file takes precedence; retain this label only as metadata.
    set_full_update_environment!("FU_INIT_KIND", "loaded")
end

include(joinpath(@__DIR__, "Full_update_J1_SU2_cell.jl"))
