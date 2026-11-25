using TOML, JLD2, Printf
const RESULTS = normpath(joinpath(@__DIR__,"..","..","results"))
const OUTPUT = joinpath(RESULTS,"parameter_scan_D4")
points = []
for ratio in (0.0,0.1,0.2,0.4,0.6)
    directory = ratio==0.2 ? joinpath(RESULTS,"gapped_D4") : joinpath(OUTPUT,"r"*replace(string(ratio),"."=>"p"))
    all(isfile(joinpath(directory,"$(method)_measurements.toml")) for method in ("su","fu","ad")) || continue
    exact = ratio==0.2 ? TOML.parsefile(joinpath(directory,"exact.toml"))["reference"][end] :
                        TOML.parsefile(joinpath(directory,"settings.toml"))["reference"][end]
    point = Dict{String,Any}("ratio"=>ratio,"mu"=>exact["mu"],"gap"=>exact["band_gap"],
                            "exact_energy"=>exact["energy_per_site"],"source"=>directory)
    for method in ("su","fu","ad")
        selected = method=="ad" && isfile(joinpath(directory,"ad_strict_measurements.toml")) ? "ad_strict" : method
        final_file = joinpath(OUTPUT,"r"*replace(string(ratio),"."=>"p"),"$(method)_final_measurements.toml")
        initial_measurements = TOML.parsefile(joinpath(directory,"$(selected)_measurements.toml"))["measurements"]
        strict_measurements = isfile(final_file) ? TOML.parsefile(final_file)["measurements"] : []
        measurements = vcat(initial_measurements,strict_measurements)
        converged = filter(m -> m["ctm_converged"],measurements)
        isempty(converged) && error("no converged measurement for r=$ratio method=$method")
        record = deepcopy(converged[argmax([(m["chi"],-get(m,"projector_tol",1e-8)) for m in converged])])
        record["all_measurements_converged"] = length(converged)==length(measurements)
        record["selected_checkpoint"] = selected
        record["absolute_error"] = record["energy"]-point["exact_energy"]
        record["relative_error_percent"] = 100abs(record["absolute_error"])/abs(point["exact_energy"])
        record["measurement_projector_tol"] = get(record,"projector_tol",1e-8)
        record["measurement_energy_spread"] = maximum(m["energy"] for m in measurements)-minimum(m["energy"] for m in measurements)
        if !isempty(strict_measurements)
            record["strict_probe_energy_spread"] = maximum(m["energy"] for m in strict_measurements)-minimum(m["energy"] for m in strict_measurements)
        end
        same_tol = filter(m -> get(m,"projector_tol",1e-8)==record["measurement_projector_tol"],measurements)
        record["chi_energy_spread"] = maximum(m["energy"] for m in same_tol)-minimum(m["energy"] for m in same_tol)
        metadata = jldopen(joinpath(directory,"$selected.jld2"),"r") do file
            file["metadata"]
        end
        haskey(metadata,"sectors") && (record["sectors"]=metadata["sectors"])
        if method=="ad"
            record["status"] = metadata["status"]
            record["iterations"] = length(metadata["history"])-1
            record["gradient_norm"] = last(metadata["history"]).gradient_norm
            record["optimization_chi"] = metadata["chi"]
            record["objective_ctm_sweeps"] = get(metadata,"ctm_iterations",get(metadata,"iterations",0))
            record["optimization_projector_tol"] = metadata["projector_tol"]
            if selected!="ad"
                first_run = TOML.parsefile(joinpath(directory,"ad_measurements.toml"))["measurements"]
                record["first_run_measured_energy"] = first_run[argmax([m["chi"] for m in first_run])]["energy"]
            end
        elseif method=="fu"
            history = metadata["history"]
            record["recorded_local_updates"] = length(history)
            record["unconverged_local_environments"] = count(r -> !r.environment.ctm_converged,history)
            record["maximum_metric_extraction_error"] = maximum(r.environment.extraction_error for r in history)
        end
        point[method] = record
    end
    for method in ("fu","ad")
        point[method]["energy_gain_from_su"] = point["su"]["energy"]-point[method]["energy"]
    end
    push!(points,point)
end
controls = Dict{String,Any}()
extended_file = joinpath(OUTPUT,"r0p4","fu_extended_fine_final_measurements.toml")
if isfile(extended_file)
    extended_measurements = TOML.parsefile(extended_file)["measurements"]
    extended_valid = filter(m -> m["ctm_converged"],extended_measurements)
    extended_chosen = deepcopy(extended_valid[argmax([(m["chi"],-m["projector_tol"]) for m in extended_valid])])
    extended_chosen["total_fu_tau_from_su"] = 1.0
    extended_chosen["final_dt"] = 0.005
    extended_chosen["extra_als_sweeps"] = 10
    controls["extended_fu_ratio_0p4"] = extended_chosen
end
initializations = []
for label in ("seed1","seed42","seed889","trimer")
    file = joinpath(OUTPUT,"initialization",label,"su_measurements.toml")
    isfile(file) || continue
    init_measurements = TOML.parsefile(file)["measurements"]
    init_chosen = deepcopy(init_measurements[argmax([m["chi"] for m in init_measurements])])
    init_chosen["initialization"] = label
    init_chosen["measurement_projector_tol"] = 1e-8
    push!(initializations,init_chosen)
end
controls["alternate_initializations_ratio_0p2"] = initializations
open(joinpath(OUTPUT,"comparison.toml"),"w") do io
    TOML.print(io,Dict("D_max"=>4,"unit_cell"=>[1,1],"U"=>0.0,"t_up"=>1.0,
        "energy_convention"=>"per physical site, including -mu*n; mu=-(1+ratio)/2",
        "comparison"=>"FU and AD independently start from SU; finite runs, not global D=4 optima",
        "points"=>points,"controls"=>controls))
end
println("ratio    gap           exact               SU               FU               AD          SU-FU          SU-AD")
for p in points
    @printf("%4.1f    %4.1f   % .10f   % .10f   % .10f   % .10f   %.5e   %.5e\n",
        p["ratio"],p["gap"],p["exact_energy"],p["su"]["energy"],p["fu"]["energy"],p["ad"]["energy"],
        p["fu"]["energy_gain_from_su"],p["ad"]["energy_gain_from_su"])
end
