using TOML, JLD2
const OUTPUT = normpath(joinpath(@__DIR__,"..","..","results","gapped_D4"))
reference = TOML.parsefile(joinpath(OUTPUT,"exact.toml"))["reference"][end]
records = []
for method in ("su","fu","ad")
    filename = joinpath(OUTPUT,"$(method)_measurements.toml")
    isfile(filename) || error("missing final measurement for $method")
    data = TOML.parsefile(filename)["measurements"]
    best = deepcopy(data[argmax([m["chi"] for m in data])])
    best["energy_error"] = best["energy"]-reference["energy_per_site"]
    best["chi_energy_spread"] = maximum(m["energy"] for m in data)-minimum(m["energy"] for m in data)
    metadata = jldopen(joinpath(OUTPUT,"$(method).jld2"),"r") do file
        file["metadata"]
    end
    if method=="ad"
        best["optimizer_status"] = metadata["status"]
        best["optimizer_iterations"] = length(metadata["history"])-1
        best["gradient_norm"] = last(metadata["history"]).gradient_norm
        best["optimization_chi"] = metadata["chi"]
        best["ctm_sweeps_in_objective"] = metadata["iterations"]
        best["projector_tol"] = metadata["projector_tol"]
    elseif method=="fu"
        best["optimization_chi"] = metadata["chi"]
        best["last_run_local_environments_converged"] = all(r.environment.ctm_converged for r in metadata["history"])
        best["last_run_maximum_metric_extraction_error"] = maximum(r.environment.extraction_error for r in metadata["history"])
    end
    push!(records,best)
    println("$method: E=$(best["energy"]) error=$(best["energy_error"]) n=$(best["density"]) chi=$(best["chi"]) CTM_converged=$(best["ctm_converged"])")
end
settings = Dict("seed"=>888,
                "su_schedule_tau_dt"=>[[5.0,0.1],[3.0,0.02],[1.0,0.005],[0.2,0.001],[2.0,0.005],[0.5,0.001]],
                "fu_additional_schedule_tau_dt"=>[[0.1,0.05],[0.2,0.02],[0.05,0.005]],
                "fu_als_sweeps_by_stage"=>[4,6,6],
                "fu_projector_tol_by_stage"=>[1e-10,1e-6,1e-6],
                "ad_maxiter"=>50,"ad_initial_step_size"=>1.0,"ad_lbfgs_memory"=>8,
                "initialization"=>"FU and AD independently start from the same SU checkpoint",
                "su_up_bond_sectors_even_odd"=>[[2,2],[2,2],[2,2]],
                "su_down_bond_sectors_even_odd"=>[[3,1],[1,3],[3,1]],
                "ad_sector_dimensions"=>"fixed to the SU checkpoint; D is the total even+odd dimension",
                "energy_convention"=>"per physical site, including -mu*n",
                "scope"=>"single initialization and unit cell; not a proof of the global D=4 optimum",
                "superseded_results"=>"pre_fix/ contains invalid measurements from before the fermionic blocking correction")
open(io -> TOML.print(io,Dict("reference"=>reference,"D_max"=>4,"unit_cell"=>[1,1],"run_settings"=>settings,"results"=>records)),
     joinpath(OUTPUT,"comparison.toml"),"w")
