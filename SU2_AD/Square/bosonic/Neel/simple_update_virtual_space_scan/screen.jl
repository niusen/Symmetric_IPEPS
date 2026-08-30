include("scan_lib.jl")

parse_list(name, default, parser=identity) =
    [parser(strip(value)) for value in split(get(ENV, name, default), ',') if !isempty(strip(value))]

const QUICK_SCHEDULES = Dict(
    "anneal_short" => [
        (dt=0.10, tau=1.0),
        (dt=0.05, tau=0.5),
        (dt=0.02, tau=0.4),
        (dt=0.01, tau=0.2),
    ],
    "anneal_fine" => [
        (dt=0.05, tau=1.0),
        (dt=0.02, tau=0.6),
        (dt=0.01, tau=0.4),
        (dt=0.005, tau=0.2),
    ],
    "legacy_long" => [
        (dt=0.10, tau=30.0),
        (dt=0.05, tau=20.0),
        (dt=0.01, tau=20.0),
        (dt=0.002, tau=4.0),
    ],
)

init_names = parse_list(
    "SCAN_INITS",
    "mixed_min,mixed_balanced,mixed_broad,paper_union," *
    "paper_x_columnar,paper_x_staggered,paper_y_columnar,paper_y_staggered",
    Symbol,
)
seeds = parse_list("SCAN_SEEDS", "666", value -> parse(Int, value))
schedule_names = parse_list("SCAN_SCHEDULES", "anneal_short")
chi = parse(Int, get(ENV, "SCAN_CHI", "16"))
ctm_tolerance = parse(Float64, get(ENV, "SCAN_CTM_TOL", "1e-5"))
ctm_maxiter = parse(Int, get(ENV, "SCAN_CTM_MAXITER", "80"))
measure_ctm = !parse(Bool, get(ENV, "SCAN_SKIP_CTM", "false"))
save_stages = parse(Bool, get(ENV, "SCAN_SAVE_STAGES", "false"))
run_id = get(ENV, "SCAN_RUN_ID", Dates.format(now(), "yyyymmdd_HHMMSS"))
run_dir = joinpath(SCAN_DIR, "results", run_id)
mkpath(run_dir)
summary_file = joinpath(run_dir, "summary.csv")

open(joinpath(run_dir, "config.txt"), "w") do io
    println(io, "run_id=$run_id")
    println(io, "inits=$(join(init_names, ','))")
    println(io, "seeds=$(join(seeds, ','))")
    println(io, "schedules=$(join(schedule_names, ','))")
    println(io, "chi=$chi")
    println(io, "ctm_tolerance=$ctm_tolerance")
    println(io, "ctm_maxiter=$ctm_maxiter")
    println(io, "measure_ctm=$measure_ctm")
    println(io, "save_stages=$save_stages")
    println(io, "paper_energy=$PAPER_ENERGY_SU2_DSTAR4")
    println(io, "qmc_energy=$QMC_ENERGY")
end

println("Results: $run_dir")
println("Paper SU(2) D*=4 reference: $PAPER_ENERGY_SU2_DSTAR4")

rows = NamedTuple[]
for schedule_name in schedule_names
    haskey(QUICK_SCHEDULES, schedule_name) || error("unknown schedule $schedule_name")
    schedule = QUICK_SCHEDULES[schedule_name]
    for init_kind in init_names, seed in seeds
        row = scan_run_case(
            run_dir,
            init_kind,
            seed,
            schedule_name,
            schedule;
            chi,
            ctm_tolerance,
            ctm_maxiter,
            measure_ctm,
        )
        push!(rows, row)
        scan_append_csv(summary_file, row)
    end
end

successful = filter(row -> row.status == "ok", rows)
sort!(successful; by=row -> isfinite(row.energy) ? row.energy : row.su_energy)
println("\n=== ranking ===")
for (rank, row) in pairs(successful)
    println(
        "$rank. $(row.case): SU E=$(row.su_energy), CTM E=$(row.energy), " *
        "Δpaper=$(row.energy - PAPER_ENERGY_SU2_DSTAR4)",
    )
end
