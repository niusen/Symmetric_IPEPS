include("scan_lib.jl")

length(ARGS) >= 1 || error("usage: julia measure_saved.jl STATE.jld2 [CHI]")
state_file = abspath(ARGS[1])
chi_value = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 24
data = load(state_file)
T_set = data["T_set"]
if haskey(data, "lambda_x") && haskey(data, "lambda_y")
    simple_measurement = scan_simple_energy(T_set, data["lambda_x"], data["lambda_y"])
    println("Simple-Update E/site=$(simple_measurement.energy_per_site)")
    println("Simple-Update Ex=$(simple_measurement.Ex)")
    println("Simple-Update Ey=$(simple_measurement.Ey)")
end
parse(Bool, get(ENV, "SCAN_SKIP_CTM", "false")) && exit()
measurement = scan_energy(
    T_set,
    chi_value;
    tolerance=parse(Float64, get(ENV, "SCAN_CTM_TOL", "1e-6")),
    maxiter=parse(Int, get(ENV, "SCAN_CTM_MAXITER", "120")),
    verbose=parse(Bool, get(ENV, "SCAN_CTM_VERBOSE", "true")),
)
println("state=$state_file")
println("E/site=$(measurement.energy_per_site)")
println("Ex=$(measurement.Ex)")
println("Ey=$(measurement.Ey)")
println("chi=$(measurement.chi), CTM iterations=$(measurement.ctm_iterations), error=$(measurement.ctm_error)")
