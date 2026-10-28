# Configuration entry support; all work stays inside Julia.
function run_native_conversion(cfg)
    isempty(cfg.input) && error("fill conversion_options.input in the Julia settings file")
    ispath(cfg.out) && error("choose a new conversion output folder")
    data=JLD2.load(cfg.input)
    source=data["format"]=="kagome_su2_charged_sqrt3_v1" ? first(KagomeChargedSU2.load_charged_checkpoint(cfg.input)) :
        data["format"]=="kagome_u1_cells_v1" ? first(KagomeU1Cells.load_u1_checkpoint(cfg.input)) :
        data["format"]=="kagome_dense_native_v1" ? data["state"] : error("unsupported conversion checkpoint")
    state=conversion_state(source;representation=cfg.representation,component=cfg.component)
    mkpath(cfg.out);save_converted_state(joinpath(cfg.out,"$(cfg.representation)_state.jld2"),state;h=cfg.h)
    if cfg.representation==:u1
        atomic_toml(joinpath(cfg.out,"u1_state.toml"),KagomeU1Cells.export_u1_portable(state))
    end
    if get(cfg,:measure,true)
        return measure_checkpoint(cfg.input,joinpath(cfg.out,"ctm");representation=cfg.representation,
            component=cfg.component,h=cfg.h,chi=cfg.chi,iterations=cfg.ctm_iterations,
            tolerance=cfg.ctm_tolerance,device=cfg.device,threads=cfg.threads)
    end
    return state
end
