using TOML, JLD2, Printf
const OUTPUT = normpath(joinpath(@__DIR__,"..","..","results","pairing_D4"))
points=[]
for delta in (0.5,1.0,2.0)
    directory=joinpath(OUTPUT,"delta"*replace(string(delta),"."=>"p"))
    all(isfile(joinpath(directory,"$(m)_measurements.toml")) for m in ("su","fu","ad")) || continue
    settings=TOML.parsefile(joinpath(directory,"settings.toml"))
    exact=settings["reference"][end]
    point=Dict{String,Any}("delta"=>delta,"reference"=>exact)
    for method in ("su","fu","ad")
        rows=TOML.parsefile(joinpath(directory,"$(method)_measurements.toml"))["measurements"]
        valid=filter(r -> r["ctm_converged"],rows)
        isempty(valid) && error("no converged measurement delta=$delta $method")
        row=deepcopy(valid[argmax([(r["chi"],-r["projector_tol"]) for r in valid])])
        row["measurement_energy_spread"]=maximum(r["energy"] for r in rows)-minimum(r["energy"] for r in rows)
        row["all_measurements_converged"]=all(r["ctm_converged"] for r in rows)
        row["density_error"]=row["density"]-exact["density"]
        row["pair_real_error"]=row["pair_real"]-exact["pair_real"]
        @assert row["energy_error"] > -1e-7 "energy below exact reference"
        @assert 0 <= row["density"] <= 2
        @assert -1e-8 <= row["double_occupancy"] <= row["density"]/2+1e-8
        @assert row["pair_real"]^2+row["pair_imag"]^2 <= row["double_occupancy"]*(1-row["density"]+row["double_occupancy"])+1e-8
        obfile=joinpath(directory,"$(method)_chi$(row["chi"])_tol$(row["projector_tol"])_ob.jld2")
        observables=jldopen(obfile,"r") do f; f["observables"]; end
        onsite=row["density"]+2delta*row["pair_real"]
        row["kinetic_up"]=real(sum(observables.up_with_onsite))/length(observables.n)-onsite
        row["kinetic_down"]=sum(observables.down_kinetic)/length(observables.n)
        row["kinetic_orientation_difference"]=row["kinetic_up"]-row["kinetic_down"]
        @assert isapprox(row["energy"],row["kinetic_up"]+row["kinetic_down"]+onsite;atol=1e-10)
        metadata=jldopen(joinpath(directory,"$method.jld2"),"r") do f; f["metadata"]; end
        row["sectors"]=metadata["sectors"]
        if method=="ad"
            row["status"]=metadata["status"]
            row["iterations"]=length(metadata["history"])-1
            row["gradient_norm"]=last(metadata["history"]).gradient_norm
            row["last_10_iterations_energy_drop"]=metadata["history"][max(1,length(metadata["history"])-10)].energy-last(metadata["history"]).energy
            row["objective_energy"]=last(metadata["history"]).energy
            row["objective_minus_measurement"]=row["objective_energy"]-row["energy"]
            row["optimization_chi"]=metadata["chi"]
            row["objective_ctm_sweeps"]=metadata["ctm_iterations"]
            row["optimization_projector_tol"]=metadata["projector_tol"]
            row["initial_objective_energy_tolerance"]=get(metadata,"initial_objective_energy_tolerance",2e-5)
            row["initial_gradient_check"]=last(TOML.parsefile(joinpath(directory,"ad_initial_checks.toml"))["attempts"])
            row["final_gradient_check"]=TOML.parsefile(joinpath(directory,"ad_final_gradient.toml"))
        elseif method=="fu"
            row["unconverged_local_environments"]=count(r -> !r.environment.ctm_converged,metadata["history"])
            row["maximum_metric_extraction_error"]=maximum(r.environment.extraction_error for r in metadata["history"])
        else
            row["last_stage_energy_change"]=last(metadata["history"]).energy_change
            row["total_tau"]=sum(r.tau for r in metadata["history"])
        end
        point[method]=row
    end
    for method in ("fu","ad")
        point[method]["energy_gain_from_su"]=point["su"]["energy"]-point[method]["energy"]
    end
    measurements=Dict(m=>TOML.parsefile(joinpath(directory,"$(m)_measurements.toml"))["measurements"] for m in ("su","fu","ad"))
    gains=[]
    for surow in measurements["su"]
        surow["ctm_converged"] || continue
        record=Dict{String,Any}("chi"=>surow["chi"],"projector_tol"=>surow["projector_tol"])
        for method in ("fu","ad")
            matches=filter(r -> r["ctm_converged"] && r["chi"]==surow["chi"] && r["projector_tol"]==surow["projector_tol"],measurements[method])
            isempty(matches) || (record["su_minus_$(method)"]=surow["energy"]-only(matches)["energy"])
        end
        push!(gains,record)
    end
    point["matched_measurement_energy_gains"]=gains
    point["reflection_checks"]=TOML.parsefile(joinpath(directory,"reflection_validation.toml"))["checks"]
    point["reference_mesh_energy_difference"]=settings["reference"][end]["energy_per_site"]-settings["reference"][1]["energy_per_site"]
    push!(points,point)
end
open(joinpath(OUTPUT,"comparison.toml"),"w") do io
    TOML.print(io,Dict("t"=>1.0,"mu"=>-1.0,"U"=>0.0,"D_max"=>4,"unit_cell"=>[1,1],
        "energy_convention"=>"per physical site, including -mu*n and explicit pairing source",
        "pair_convention"=>"F=<c_down*c_up>, H_pair=Delta*c_up†*c_down†+conj(Delta)*c_down*c_up",
        "points"=>points))
end
println("Delta         exact              SU              FU              AD          SU-FU         SU-AD")
for p in points
    @printf("%4.1f   % .10f  % .10f  % .10f  % .10f  %.4e  %.4e\n",p["delta"],p["reference"]["energy_per_site"],
        p["su"]["energy"],p["fu"]["energy"],p["ad"]["energy"],p["fu"]["energy_gain_from_su"],p["ad"]["energy_gain_from_su"])
end
