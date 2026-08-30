include("scan_lib.jl")

length(ARGS) >= 1 || error("usage: julia continue_saved.jl STATE.jld2 [OUTPUT_NAME]")

const CONTINUE_SCHEDULES = Dict(
    "legacy_long" => [
        (dt=0.10, tau=30.0),
        (dt=0.05, tau=20.0),
        (dt=0.01, tau=20.0),
        (dt=0.002, tau=4.0),
    ],
    "fine_long" => [
        (dt=0.002, tau=10.0),
        (dt=0.001, tau=6.0),
        (dt=0.0005, tau=3.0),
        (dt=0.0002, tau=1.0),
    ],
    "fine_medium" => [
        (dt=0.002, tau=4.0),
        (dt=0.001, tau=2.0),
        (dt=0.0005, tau=1.0),
    ],
)

state_file = abspath(ARGS[1])
schedule_name = get(ENV, "SCAN_CONT_SCHEDULE", "fine_long")
haskey(CONTINUE_SCHEDULES, schedule_name) ||
    error("unknown continuation schedule $schedule_name")
schedule = CONTINUE_SCHEDULES[schedule_name]
output_name = length(ARGS) >= 2 ? ARGS[2] :
    "continue_$(schedule_name)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
output_dir = joinpath(SCAN_DIR, "results", output_name)
mkpath(output_dir)

data = load(state_file)
T_set = data["T_set"]
lambda_x = data["lambda_x"]
lambda_y = data["lambda_y"]

open(joinpath(output_dir, "config.txt"), "w") do io
    println(io, "source_state=$state_file")
    println(io, "schedule=$schedule_name")
    println(io, "Dstar=$(get(ENV, "SCAN_DSTAR", "4"))")
    println(io, "Dmax=$(get(ENV, "SCAN_DMAX", "12"))")
    println(io, "multiplet_tolerance=$(get(ENV, "SCAN_MULTIPLET_TOL", "1e-5"))")
    println(io, "save_stages=$(get(ENV, "SCAN_SAVE_STAGES", "false"))")
end

initial_report = square_J1_bond_space_report(lambda_x, lambda_y)
jldsave(joinpath(output_dir, "initial.jld2"); T_set, lambda_x, lambda_y, initial_report)

T_set, lambda_x, lambda_y, stage_records = scan_run_schedule!(
    T_set,
    lambda_x,
    lambda_y,
    schedule,
    output_dir;
    Dstar=parse(Int, get(ENV, "SCAN_DSTAR", "4")),
    Dmax=parse(Int, get(ENV, "SCAN_DMAX", "12")),
    multiplet_tolerance=parse(Float64, get(ENV, "SCAN_MULTIPLET_TOL", "1e-5")),
)

simple_measurement = scan_simple_energy(T_set, lambda_x, lambda_y)
final_report = square_J1_bond_space_report(lambda_x, lambda_y)
jldsave(
    joinpath(output_dir, "final.jld2");
    T_set,
    lambda_x,
    lambda_y,
    source_state=state_file,
    schedule_name,
    schedule,
    stage_records,
    final_report,
    simple_measurement,
)

println("output=$output_dir")
println("SU E/site=$(simple_measurement.energy_per_site)")
square_J1_print_bond_spaces(lambda_x, lambda_y; prefix="  ")
