const _LEGACY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function _include_if_needed(path::AbstractString)
    isfile(path) || error("legacy file not found: $path")
    Base.include(Main, path)
    return nothing
end

"""
    load_legacy_spin_hall!()

Load the existing CTMRG, PESS, and spin-Hall observable code from `SU2_AD/src`.
This keeps the new graded code in `auto_fermionic`, while reusing the tested
environment contraction routines.
"""
function load_legacy_spin_hall!()
    root = _LEGACY_ROOT
    for rel in (
        "bosonic/Settings.jl",
        "bosonic/Settings_cell.jl",
        "bosonic/iPEPS_ansatz.jl",
        "bosonic/AD_lib.jl",
        "bosonic/CTMRG.jl",
        "fermionic/Fermionic_CTMRG.jl",
        "fermionic/Fermionic_CTMRG_unitcell.jl",
        "fermionic/square_Hubbard_model_cell.jl",
        "fermionic/swap_funs.jl",
        "fermionic/fermi_permute.jl",
        "fermionic/mpo_mps_funs.jl",
        "fermionic/double_layer_funs.jl",
        "fermionic/square_Hubbard_AD_cell.jl",
        "fermionic/triangle_fiPESS_method.jl",
    )
        _include_if_needed(joinpath(root, rel))
    end
    return nothing
end

"""
    evaluate_legacy_spin_hall(parameters, A_cell, AA_cell, CTM_cell, ctm_setting, energy_setting)

Thin wrapper around the old `evaluate_ob_cell`. Use this after constructing
`A_cell`, `AA_cell`, and `CTM_cell` with the existing CTMRG workflow.
"""
function evaluate_legacy_spin_hall(parameters, A_cell, AA_cell, CTM_cell, ctm_setting, energy_setting)
    isdefined(Main, :evaluate_ob_cell) || load_legacy_spin_hall!()
    return Main.evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, ctm_setting, energy_setting)
end

