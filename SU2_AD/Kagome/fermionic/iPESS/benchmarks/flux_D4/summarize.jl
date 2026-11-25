using TOML,JLD2,Printf
const BASE=normpath(joinpath(@__DIR__,"..","..","results","flux_D4"))
const OUT=isempty(ARGS) ? BASE : joinpath(BASE,only(ARGS))
settings=TOML.parsefile(joinpath(OUT,"settings.toml")); exact=settings["reference"]
result=Dict{String,Any}("reference"=>exact,"D_max"=>4,"seed"=>settings["seed"],"spin_degeneracy"=>2,"methods"=>Dict(),"matched_gains"=>[])
measurements=Dict()
for method in ("su","fu","ad")
    rows=TOML.parsefile(joinpath(OUT,"$(method)_measurements.toml"))["measurements"]
    measurements[method]=rows
    valid=filter(r->r["ctm_converged"],rows); isempty(valid) && error("no converged $method measurement")
    r=deepcopy(valid[argmax([(r["chi"],-r["projector_tol"]) for r in valid])])
    r["kinetic_energy"]=r["energy"]+settings["mu"]*r["density"]
    r["density_error"]=r["density"]-exact["density"]
    r["energy_setting_spread"]=maximum(x["energy"] for x in valid)-minimum(x["energy"] for x in valid)
    r["all_measurements_converged"]=all(x["ctm_converged"] for x in rows)
    ob=jldopen(joinpath(OUT,"$(method)_chi$(r["chi"])_tol$(r["projector_tol"])_ob.jld2"),"r") do f; f["observables"]; end
    r["site_densities"]=real.(vec(ob.n))
    @assert r["energy_error"] > -1e-7
    @assert 0<=r["density"]<=2
    @assert r["pair_real"]^2+r["pair_imag"]^2 <= r["double_occupancy"]*(1-r["density"]+r["double_occupancy"])+1e-8
    metadata=jldopen(joinpath(OUT,"$method.jld2"),"r") do f; f["metadata"]; end
    r["sectors"]=metadata["sectors"]
    if method=="ad"
        r["iterations"]=length(metadata["history"])-1
        r["status"]=metadata["status"]
        r["gradient_norm"]=last(metadata["history"]).gradient_norm
        r["last10_energy_drop"]=metadata["history"][max(1,end-10)].energy-last(metadata["history"]).energy
        r["objective_minus_measurement"]=last(metadata["history"]).energy-r["energy"]
        r["optimization_chi"]=metadata["chi"]
        r["optimization_projector_tol"]=metadata["projector_tol"]
        r["objective_ctm_sweeps"]=metadata["ctm_iterations"]
        r["final_gradient_check"]=TOML.parsefile(joinpath(OUT,"ad_final_gradient.toml"))
        gc=r["final_gradient_check"]
        r["final_gradient_passed"]=gc["relative_error"]<1e-3 || gc["absolute_error"]<1e-7
        for name in ("ad_tighten_checks","ad_chi_checks")
            probe_file=joinpath(OUT,"$name.toml")
            isfile(probe_file) && (r[name]=TOML.parsefile(probe_file)["checks"])
        end
        auditfile=joinpath(OUT,"ad_audit.toml")
        if isfile(auditfile)
            audit=TOML.parsefile(auditfile)["checks"]
            r["gradient_audit"]=audit
            r["all_audit_directions_passed"]=all(x["passed"] for x in audit)
            r["objective_sweep_energy_spread"]=maximum(x["energy"] for x in audit)-minimum(x["energy"] for x in audit)
        end
        r["history"]=[Dict(string(k)=>v for (k,v) in pairs(x)) for x in metadata["history"]]
        budget=[]
        for i in (50,100)
            file=joinpath(OUT,"ad_iter$(i)_measurements.toml")
            isfile(file) && push!(budget,merge(Dict("iteration"=>i),only(TOML.parsefile(file)["measurements"])))
        end
        r["budget_comparison"]=budget
    elseif method=="fu"
        r["stages"]=[Dict(string(k)=>v for (k,v) in pairs(s)) for s in metadata["stages"]]
        r["total_tau"]=sum(s.tau for s in metadata["stages"])
        stages=metadata["stages"]
        r["stage_energy_converged"]=length(stages)>=4 && stages[end].dt==stages[end-1].dt && abs(last(stages).energy_change)<2e-7
        r["local_updates"]=length(metadata["history"])
        r["unconverged_local_environments"]=count(x->!x.environment.ctm_converged,metadata["history"])
        r["max_metric_extraction_error"]=maximum(x.environment.extraction_error for x in metadata["history"])
        r["max_metric_hermitian_error"]=maximum(x.environment.hermitian_error for x in metadata["history"])
        r["max_metric_negative_weight"]=maximum(x.environment.negative_weight for x in metadata["history"])
        for initial in ("su","fu")
            auditfile=joinpath(OUT,"fu_audit_$initial.toml")
            isfile(auditfile) && (r["local_fit_audit_$initial"]=TOML.parsefile(auditfile))
        end
        bestfile=joinpath(OUT,"fu_best_stage_measurements.toml")
        if isfile(bestfile)
            r["best_stage_measurement"]=only(TOML.parsefile(bestfile)["measurements"])
            r["best_stage_selection"]=TOML.parsefile(joinpath(OUT,"fu_best_stage_selection.toml"))
        end
    else
        r["total_tau"]=sum(s.tau for s in metadata["history"])
        r["last_stage_energy_change"]=last(metadata["history"]).energy_change
    end
    result["methods"][method]=r
    @printf("%s E=% .12f error=%.6e kinetic=% .12f n=%.9f\n",uppercase(method),r["energy"],r["energy_error"],r["kinetic_energy"],r["density"])
end
for method in ("fu","ad")
    result["methods"][method]["gain_from_su"]=result["methods"]["su"]["energy"]-result["methods"][method]["energy"]
end
for sr in measurements["su"]
    sr["ctm_converged"] || continue
    row=Dict{String,Any}("chi"=>sr["chi"],"projector_tol"=>sr["projector_tol"])
    for m in ("fu","ad")
        matches=filter(r->r["ctm_converged"] && r["chi"]==sr["chi"] && r["projector_tol"]==sr["projector_tol"],measurements[m])
        isempty(matches) || (row["su_minus_$m"]=sr["energy"]-only(matches)["energy"])
    end
    push!(result["matched_gains"],row)
end
result["validation"]=TOML.parsefile(joinpath(OUT,"validation.toml"))["checks"]
seed_checks=[]
for seed in (777,888,999)
    file=joinpath(seed==888 ? BASE : joinpath(BASE,"seed$seed"),"su_measurements.toml")
    if isfile(file)
        rows=filter(r->r["ctm_converged"],TOML.parsefile(file)["measurements"])
        best=rows[argmax([(r["chi"],-r["projector_tol"]) for r in rows])]
        push!(seed_checks,merge(Dict("seed"=>seed),best))
    end
end
result["initialization_checks"]=seed_checks
open(io->TOML.print(io,result),joinpath(OUT,"comparison.toml"),"w")
