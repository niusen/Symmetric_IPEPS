include("scan_lib.jl")

length(ARGS) == 1 || error("usage: julia analyze_saved_series.jl CASE_DIRECTORY")
case_dir = abspath(ARGS[1])
isdir(case_dir) || error("not a directory: $case_dir")

stage_number(path) = begin
    match_result = match(r"stage_(\d+)_", basename(path))
    isnothing(match_result) ? typemax(Int) : parse(Int, match_result.captures[1])
end
state_files = sort(
    filter(path -> occursin(r"^stage_\d+_.*\.jld2$", basename(path)),
           readdir(case_dir; join=true));
    by=stage_number,
)
isfile(joinpath(case_dir, "final.jld2")) && push!(state_files, joinpath(case_dir, "final.jld2"))
isempty(state_files) && error("no stage or final states found in $case_dir")

csv_file = joinpath(case_dir, "stage_energy.csv")
report_file = joinpath(case_dir, "lambda_report.txt")

open(csv_file, "w") do csv
    println(csv, "state,su_energy,Ex_mean,Ey_mean")
    open(report_file, "w") do report_io
        for state_file in state_files
            data = load(state_file)
            T_set = data["T_set"]
            lambda_x = data["lambda_x"]
            lambda_y = data["lambda_y"]
            measurement = scan_simple_energy(T_set, lambda_x, lambda_y)
            label = basename(state_file)
            println(
                csv,
                join((label, measurement.energy_per_site,
                      sum(measurement.Ex) / length(measurement.Ex),
                      sum(measurement.Ey) / length(measurement.Ey)), ','),
            )
            println(report_io, "=== $label ===")
            println(report_io, "SU E/site=$(measurement.energy_per_site)")
            for bond in square_J1_bond_space_report(lambda_x, lambda_y)
                println(
                    report_io,
                    bond.direction,
                    bond.from,
                    "→",
                    bond.to,
                    ": ",
                    bond.space,
                    " parity=",
                    bond.parity,
                    " lambda={",
                    _square_su_format_lambda(bond.lambda),
                    "}",
                )
            end
            println(report_io)
            println("$label: SU E/site=$(measurement.energy_per_site)")
        end
    end
end

println("energy_csv=$csv_file")
println("lambda_report=$report_file")
