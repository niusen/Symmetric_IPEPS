# All settings are in this file. This measures a saved U1 state; no SU runs.
# First use small D; do not replace the source with D15 before the chi/L checks.
import Pkg
const VARIANCE_CELL_ROOT=normpath(joinpath(@__DIR__,".."))
if isnothing(Base.active_project()) || normpath(Base.active_project())!=joinpath(VARIANCE_CELL_ROOT,"Project.toml")
    Pkg.activate(VARIANCE_CELL_ROOT)
end
const VARIANCE_LIBRARY=joinpath(VARIANCE_CELL_ROOT,"..","..","..","src","bosonic","kagome",
                              "native_measurement","energy_variance","KagomeEnergyVariance.jl")
include(VARIANCE_LIBRARY)
using .KagomeEnergyVariance

variance_settings=(
    checkpoint=joinpath(VARIANCE_CELL_ROOT,"results","u1_from_D4_D4_D10_h038_chi40","D4","final.jld2"),
    out=joinpath(VARIANCE_CELL_ROOT,"results","variance_U1_D4_L12_chi8_sweeps2"),
    threads=10,
    L=12,                        # Square contraction window, multiple of 3.
    radii=[0,1,2],                # Chebyshev radius of primitive-cell origins.
    chi=8,
    ctm_iterations=30,            # Original periodic norm environment only.
    ctm_tolerance=1e-8,
    sweeps=2,                    # Center-outward LC sweeps; check 1 vs 2 vs 3.
    source_shift=1.0,            # Source h0+I has nonzero overlap. Covariance unchanged.
)

if abspath(PROGRAM_FILE)==@__FILE__
    cfg=variance_settings
    result=measure_variance_checkpoint(cfg.checkpoint,cfg.out;threads=cfg.threads,
        L=cfg.L,radii=cfg.radii,chi=cfg.chi,ctm_iterations=cfg.ctm_iterations,
        ctm_tolerance=cfg.ctm_tolerance,sweeps=cfg.sweeps,source_shift=cfg.source_shift)
    println("Exchange energy/site: ",result["exchange_energy_per_spin"])
    for row in result["cumulative"];println(row);end
    println("Saved: ",cfg.out)
end
