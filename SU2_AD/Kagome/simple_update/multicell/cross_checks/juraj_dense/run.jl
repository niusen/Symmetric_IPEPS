# U1 input -> dense Juraj JSON -> Juraj's own dense CTMRG/RDM.
# Settings for the cross-check only; no simple update or optimization.
multicell_dir=normpath(joinpath(@__DIR__,"..",".."))
original_results=joinpath(multicell_dir,"results","u1_from_D4_D4_D10_h038_chi40")
juraj_check_settings=(
    dims=[9,10],
    chis=[20,40],
    h=.38,
    threads=10,
    device="cpu",
    ctm_iterations=50,
    ctm_tolerance=1e-6,
    projector_cutoff=1e-12,
    # Set this to the existing tn-torch_dev checkout on the running computer.
    juraj_repository=raw"D:/My Documents/Code/python_codes/Juraj/tn-torch_dev",
    source_root=original_results,
    out=joinpath(multicell_dir,"results","juraj_dense_from_u1_D9_D10"),
)
if !isdefined(@__MODULE__,:KAGOME_SETTINGS_ONLY)
    import Pkg
    Pkg.activate(multicell_dir)
    Pkg.instantiate()
    include(joinpath(multicell_dir,"..","..","..","src","bosonic","kagome","native_measurement","juraj_dense_export.jl"))
    cfg=juraj_check_settings
    isfile(joinpath(cfg.juraj_repository,"ipeps","ipess_kagome.py")) || error("set juraj_repository to the existing Juraj repository")
    mkpath(cfg.out);points=Dict[]
    for D in cfg.dims
        source=joinpath(cfg.source_root,"D$D","u1_state.toml")
        isfile(source) || error("missing final portable U1 state: $source")
        json=joinpath(cfg.out,"D$D","juraj_dense_state.json")
        KagomeJurajDenseExport.export_juraj_dense(source,json;D)
        push!(points,Dict("D"=>D,"state_json"=>abspath(json),
            "julia_reference"=>joinpath(abspath(cfg.source_root),"D$D","ctm","observables.toml")))
    end
    plan=Dict("format"=>"julia_u1_to_juraj_dense_check_v1","states"=>points,"chis"=>cfg.chis,
        "h"=>cfg.h,"threads"=>cfg.threads,"device"=>cfg.device,"ctm_iterations"=>cfg.ctm_iterations,
        "ctm_tolerance"=>cfg.ctm_tolerance,"projector_cutoff"=>cfg.projector_cutoff,
        "juraj_repository"=>abspath(cfg.juraj_repository),"out"=>abspath(cfg.out))
    KagomeNativeMeasurement.write_json(joinpath(cfg.out,"measurement_plan.json"),plan)
    # Stable launcher-side pointer avoids duplicating output settings in sh.
    write(joinpath(@__DIR__,"measurement_plan_path.txt"),abspath(joinpath(cfg.out,"measurement_plan.json")))
end
