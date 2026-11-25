include("run.jl")
directory = joinpath(OUTPUT,"r0p1")
state,parameters,_ = load_checkpoint(joinpath(directory,"su.jld2"))
v,_ = pack_state(state)
measured = TOML.parsefile(joinpath(directory,"su_measurements.toml"))["measurements"][end]["energy"]
checks = []
for tolerance in (1e-6,1e-7,1e-8)
    problem = VariationalProblem(state,parameters;chi=12,ctm_iterations=4,projector_tol=tolerance)
    report = check_gradient(v,problem;epsilon=1e-5)
    record = merge(asdict(report),Dict("projector_tol"=>tolerance,"chi"=>12,
                   "ctm_iterations"=>4,"difference_from_measurement"=>report.energy-measured))
    push!(checks,record)
    note(directory,"TOLERANCE_CHECK $record")
    write_toml(joinpath(directory,"ad_tolerance_checks.toml"),Dict("checks"=>checks))
end
