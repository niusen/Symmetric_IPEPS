# Restart orchestration only; numerical work stays in the existing run_one.
if !isdefined(@__MODULE__, :DenseSU4LongRun)
    include(joinpath(@__DIR__, "LongRun.jl"))
end
module DenseSU4Resume
using ..DenseSU4LongRun
using MAT, Dates
const LR = DenseSU4LongRun
export resume_batch

function saved_result_row(cfg, D, seed)
    path = joinpath(LR.run_directory(cfg, D, seed), "result.mat")
    isfile(path) || return nothing
    # Read metadata only, without loading exported tensors.
    fields = ("complete", "Dmax", "seed", "J", "convention", "Lx", "Ly",
        "tensor_format", "environment", "energy_per_site", "converged", "total_sweeps",
        "history", "history_columns", "initialization_mode", "source_D", "expansion_noise",
        "ctm_chi", "ctm_max_iterations", "ctm_corner_tol", "ctm_chi_tol")
    data = matopen(path) do file
        Dict(key => read(file, key) for key in fields)
    end
    data["complete"] == true || return nothing
    data["Dmax"] == D && data["seed"] == seed && data["J"] == cfg.J &&
        data["convention"] == string(cfg.convention) &&
        data["Lx"] == cfg.Lx && data["Ly"] == cfg.Ly &&
        data["tensor_format"] == "TensorMap_ComplexSpace" && data["environment"] == "ctmrg" &&
        data["initialization_mode"] == string(cfg.initialization) &&
        data["ctm_chi"] == last(cfg.ctm_chis) && data["ctm_max_iterations"] == cfg.ctm_max_iterations &&
        data["ctm_corner_tol"] == cfg.ctm_corner_tol && data["ctm_chi_tol"] == cfg.ctm_chi_tol ||
        error("Saved result parameters differ: $path")
    if LR.is_continuation(cfg, D)
        previous_D = cfg.Ds[findfirst(==(D), cfg.Ds)-1]
        data["source_D"] == previous_D && data["expansion_noise"] == cfg.expansion_noise ||
            error("Saved result continuation parameters differ: $path")
    end
    energy = Float64(data["energy_per_site"])
    isfinite(energy) || error("Non-finite energy in $path")
    columns = vec(data["history_columns"])
    elapsed_column = findfirst(==("elapsed_seconds"), columns)
    isnothing(elapsed_column) && error("Missing SU timing column in $path")
    elapsed = isempty(data["history"]) ? 0.0 : data["history"][end, elapsed_column]
    converged = data["converged"] == true
    return (D=D, seed=seed, energy=energy, converged=converged, complete=true,
        status=converged ? 1 : 2, matfile=path,
        total_sweeps=Int(data["total_sweeps"]), elapsed_seconds=Float64(elapsed))
end

"""Resume from start_D, retaining the full cfg.Ds and previous completed results."""
function resume_batch(cfg::LR.LongRunConfig=LR.LongRunConfig(); start_D=9, runner=LR.run_one)
    LR.validate(cfg)
    cfg.resume || error("Restart requires resume=true")
    start_D in cfg.Ds || error("start_D must be in the original D sequence")
    # A typo in the output root must not silently launch a new SU calculation.
    checkpoint = joinpath(LR.run_directory(cfg, start_D, first(cfg.seeds)), "checkpoint.jls")
    isfile(checkpoint) || error("Restart checkpoint not found: $checkpoint\nSet SU4_OUTPUT_DIR to the original root containing continuation/.")
    rows = Dict{Tuple{Int,Int},NamedTuple}()
    for D in cfg.Ds, seed in cfg.seeds
        row = saved_result_row(cfg, D, seed)
        isnothing(row) || (rows[(D,seed)] = row)
    end
    ordered_rows() = [rows[(D,seed)] for D in cfg.Ds for seed in cfg.seeds if haskey(rows,(D,seed))]
    LR.write_batch_summary(cfg, ordered_rows())
    println("Resume from D=", start_D, "; output=", abspath(cfg.output_dir))
    println("Retained completed results: ", length(rows), "; original D sequence: ", cfg.Ds)
    flush(stdout)
    for D in cfg.Ds, seed in cfg.seeds
        D < start_D && continue
        if haskey(rows, (D,seed))
            println("Skip completed D=", D, " seed=", seed)
            flush(stdout)
            continue
        end
        println(Dates.now(), " Starting/resuming D=", D, " seed=", seed)
        flush(stdout)
        try
            rows[(D,seed)] = runner(cfg, D, seed)
        catch err
            if !(err isa InterruptException)
                open(joinpath(LR.batch_directory(cfg), "failures.log"), "a") do io
                    println(io, Dates.now(), " D=", D, " seed=", seed, "\n",
                        sprint(showerror, err, catch_backtrace()))
                end
                rows[(D,seed)] = (D=D, seed=seed, energy=NaN, converged=false, complete=false,
                    status=-1, matfile="", total_sweeps=0, elapsed_seconds=NaN)
                LR.write_batch_summary(cfg, ordered_rows())
            end
            rethrow() # Stop on failure instead of launching larger D values.
        end
        LR.write_batch_summary(cfg, ordered_rows())
        GC.gc()
    end
    return ordered_rows()
end
end
